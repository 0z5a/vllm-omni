# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One asynchronous SeedVR2 request, bounded model windows, one MP4 writer."""

import asyncio
import hashlib
import io
import itertools
import json
import logging
import os
import shutil
import subprocess
import time
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import av
import imageio_ffmpeg
import numpy as np
import regex as re
import requests
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from vllm_omni.diffusion import envs
from vllm_omni.inputs.data import COLOR_CORRECTION_METHODS, DEFAULT_COLOR_CORRECTION_METHOD

logger = logging.getLogger(__name__)
router = APIRouter()
MAX_FRAMES = 7200
MAX_FRAME_PIXELS = 768 * 1344
FPS = 24
WINDOW, OVERLAP = 12, 4
UPLOAD_CHUNK = 1 << 20
JOB_ID = re.compile(r"[0-9a-f]{32}")
active_task: asyncio.Task[None] | None = None
job_lock = asyncio.Lock()


class JobError(Exception):
    """An error whose message is safe to return to the client."""


class JobCancelledError(JobError):
    """The client asked the job to stop."""


def _jobs_root() -> Path:
    return Path(envs.VLLM_OMNI_SEEDVR2_LONG_OUTPUT_DIR) / "seedvr2-long"


def _job_dir(job_id: str) -> Path:
    """Resolve a client-supplied job id, which must never shape the path."""
    if not JOB_ID.fullmatch(job_id):
        raise HTTPException(404, "SeedVR2 long-video job not found")
    return _jobs_root() / job_id


def _sweep_expired_jobs() -> None:
    """Drop settled job directories so repeated submissions cannot fill the disk."""
    deadline = time.time() - envs.VLLM_OMNI_SEEDVR2_LONG_JOB_TTL_SECONDS
    for job in _jobs_root().glob("*"):
        status = job / "status.json"
        if not job.is_dir() or not JOB_ID.fullmatch(job.name) or not status.exists():
            continue
        record = json.loads(status.read_text())
        # A job owned by a dead process is settled too, or a restart would leak it.
        settled = record["status"] not in {"queued", "running"} or record.get("pid") != os.getpid()
        if settled and status.stat().st_mtime <= deadline:
            shutil.rmtree(job, ignore_errors=True)


def _store_upload(upload: UploadFile, destination: Path) -> None:
    """Copy the upload under a size cap so one client cannot fill the disk."""
    limit = envs.VLLM_OMNI_SEEDVR2_LONG_MAX_UPLOAD_BYTES
    written = 0
    with destination.open("wb") as target:
        while chunk := upload.file.read(UPLOAD_CHUNK):
            written += len(chunk)
            if written > limit:
                raise HTTPException(413, f"SeedVR2 long-video upload exceeds {limit} bytes")
            target.write(chunk)
    if not written:
        raise HTTPException(400, "SeedVR2 long-video upload is empty")


def _status(job: Path, stage: str, frames: int, error: str = "") -> None:
    # The owning PID lets a poller detect a job that a server restart abandoned.
    record = {"status": stage, "frames": frames, "error": error, "pid": os.getpid()}
    pending = job / "status.json.tmp"
    pending.write_text(json.dumps(record) + "\n")
    pending.replace(job / "status.json")


def _frames(source: Path, width: int, height: int, loop: bool) -> Iterator[av.VideoFrame]:
    while True:
        count = 0
        with av.open(str(source)) as container:
            video = container.streams.video[0]
            if video.average_rate != Fraction(FPS):
                raise JobError("SeedVR2 long video requires 24 FPS input")
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None or frame.pts * frame.time_base != Fraction(count, FPS):
                    raise JobError("SeedVR2 long video requires constant 24 FPS timestamps starting at zero")
                count += 1
                yield frame.reformat(width=width, height=height, format="yuv420p", interpolation="BICUBIC")
        if count == 0:
            raise JobError("Input video has no frames")
        if not loop:
            return


def _segment(frames: list[av.VideoFrame], width: int, height: int) -> bytes:
    buffer = io.BytesIO()
    with av.open(buffer, "w", format="matroska") as container:
        stream = container.add_stream("libx264", rate=FPS)
        stream.width, stream.height, stream.pix_fmt = width, height, "yuv420p"
        stream.options = {"preset": "veryfast", "crf": "18"}
        for index, frame in enumerate(frames):
            frame.pts, frame.time_base = index, Fraction(1, FPS)
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return buffer.getvalue()


def _restore(
    frames: list[av.VideoFrame],
    width: int,
    height: int,
    seed: int,
    color_correction_method: str,
    port: int,
    authorization: str,
) -> list[np.ndarray]:
    response = requests.post(
        f"http://127.0.0.1:{port}/v1/videos/sync",
        data={
            "prompt": " ",
            "size": f"{width}x{height}",
            "num_frames": str(len(frames)),
            "num_inference_steps": "1",
            "guidance_scale": "1",
            "seed": str(seed),
            "color_correction_method": color_correction_method,
        },
        files={"input_references": ("window.mkv", _segment(frames, width, height), "video/x-matroska")},
        headers={"Authorization": authorization} if authorization else {},
        timeout=900,
    )
    try:
        response.raise_for_status()
    except requests.HTTPError as error:
        raise JobError("SeedVR2 window restoration failed") from error
    with av.open(io.BytesIO(response.content)) as container:
        video = container.streams.video[0]
        decoded = list(container.decode(video=0))
        if (video.width, video.height, video.average_rate, len(decoded)) != (width, height, Fraction(FPS), len(frames)):
            raise JobError("SeedVR2 window restoration returned an unexpected geometry or frame count")
        if [frame.pts * frame.time_base for frame in decoded] != [Fraction(index, FPS) for index in range(len(frames))]:
            raise JobError("SeedVR2 window restoration returned invalid timestamps")
        return [frame.to_ndarray(format="rgb24") for frame in decoded]


def _run(
    job: Path,
    width: int,
    height: int,
    target: int,
    loop: bool,
    seed: int,
    color_correction_method: str,
    port: int,
    authorization: str,
) -> None:
    source = job / "input.mp4"
    video_path = job / "video.mp4"
    output = job / "output.mp4"
    source_frames = _frames(source, width, height, loop)
    batch = list(itertools.islice(source_frames, min(WINDOW, target)))
    if len(batch) != min(WINDOW, target):
        raise JobError("Input video has fewer frames than requested; set loop_input=true to repeat it")
    start = written = 0
    pending: list[np.ndarray] = []
    with av.open(str(video_path), "w", format="mp4") as container:
        video = container.add_stream("libx264", rate=FPS)
        video.width, video.height, video.pix_fmt = width, height, "yuv420p"
        video.options = {"preset": "veryfast", "crf": "18", "tune": "zerolatency", "bf": "0"}

        def write(array: np.ndarray) -> None:
            nonlocal written
            frame = av.VideoFrame.from_ndarray(array, format="rgb24")
            frame.pts, frame.time_base = written, Fraction(1, FPS)
            for packet in video.encode(frame):
                container.mux(packet)
            written += 1

        while batch:
            if (job / "cancel").exists():
                raise JobCancelledError("SeedVR2 long-video job was cancelled")
            restored = _restore(batch, width, height, seed, color_correction_method, port, authorization)
            last = start + len(batch) == target
            if not pending:
                for frame in restored if last else restored[:-OVERLAP]:
                    write(frame)
            else:
                for offset in range(OVERLAP):
                    newer = (offset + 1) / (OVERLAP + 1)
                    write(np.rint(pending[offset] * (1 - newer) + restored[offset] * newer).astype(np.uint8))
                for frame in restored[OVERLAP:] if last else restored[OVERLAP:-OVERLAP]:
                    write(frame)
            _status(job, "running", written)
            if last:
                break
            pending = restored[-OVERLAP:]
            fresh = min(WINDOW - OVERLAP, target - start - len(batch))
            next_frames = list(itertools.islice(source_frames, fresh))
            if len(next_frames) != fresh:
                raise JobError("Input video has fewer frames than requested; set loop_input=true to repeat it")
            batch = batch[-OVERLAP:] + next_frames
            start += len(restored) - OVERLAP
        for packet in video.encode():
            container.mux(packet)
    if written != target:
        raise JobError(f"SeedVR2 restored {written} of {target} requested frames")

    with av.open(str(source)) as container:
        has_audio = bool(container.streams.audio)
    if has_audio:
        subprocess.run(
            [
                imageio_ffmpeg.get_ffmpeg_exe(),
                "-nostdin",
                "-y",
                "-hide_banner",
                "-loglevel",
                "error",
                "-stream_loop",
                "-1" if loop else "0",
                "-i",
                str(source),
                "-i",
                str(video_path),
                "-map",
                "1:v:0",
                "-map",
                "0:a:0",
                "-frames:v",
                str(target),
                "-t",
                str(target / FPS),
                "-c:v",
                "copy",
                "-c:a",
                "aac",
                "-b:a",
                "160k",
                "-movflags",
                "+faststart",
                str(output),
            ],
            check=True,
        )
    else:
        video_path.replace(output)
    with av.open(str(output)) as container:
        video = container.streams.video[0]
        decoded = 0
        for decoded, frame in enumerate(container.decode(video=0), start=1):
            if (
                frame.pts is None
                or frame.time_base is None
                or frame.pts * frame.time_base != Fraction(decoded - 1, FPS)
            ):
                raise JobError("SeedVR2 output encoding produced invalid timestamps")
        if (video.width, video.height, video.average_rate, decoded) != (width, height, Fraction(FPS), target):
            raise JobError("SeedVR2 output encoding failed frame or geometry validation")
        if has_audio and not container.streams.audio:
            raise JobError("SeedVR2 output encoding lost the audio track")
    if has_audio:
        with av.open(str(output)) as container:
            audio = container.streams.audio[0]
            samples = sum(frame.samples for frame in container.decode(audio=0))
            if samples < (target / FPS - 1) * audio.rate:
                raise JobError("SeedVR2 output audio ends before the video")
    video_path.unlink(missing_ok=True)
    with output.open("rb") as content:
        digest = hashlib.file_digest(content, "sha256").hexdigest()
    (job / "result.json").write_text(
        json.dumps({"frames": target, "width": width, "height": height, "fps": FPS, "sha256": digest}) + "\n"
    )
    _status(job, "completed", target)


def _background(
    job: Path,
    width: int,
    height: int,
    target: int,
    loop: bool,
    seed: int,
    color_correction_method: str,
    port: int,
    authorization: str,
) -> None:
    try:
        _run(job, width, height, target, loop, seed, color_correction_method, port, authorization)
    except Exception as error:
        logger.exception("SeedVR2 long-video job %s failed", job.name)
        current = json.loads((job / "status.json").read_text())
        detail = str(error) if isinstance(error, JobError) else "SeedVR2 long-video restoration failed"
        _status(job, "cancelled" if isinstance(error, JobCancelledError) else "failed", current["frames"], detail)


@router.post("/v1/seedvr2/restore-long", status_code=202)
async def create_long_video(
    raw_request: Request,
    input_references: UploadFile = File(...),
    num_frames: int = Form(...),
    size: str = Form(...),
    loop_input: bool = Form(False),
    prompt: str = Form(" "),
    seed: int = Form(7723),
    color_correction_method: str = Form(DEFAULT_COLOR_CORRECTION_METHOD),
) -> dict[str, str | int]:
    global active_task
    if raw_request.app.state.api_server_count != 1:
        raise HTTPException(409, "SeedVR2 long-video jobs require one API server")
    try:
        width, height = (int(value) for value in size.lower().split("x"))
    except ValueError as error:
        raise HTTPException(400, "size must be WIDTHxHEIGHT") from error
    if not 1 <= num_frames <= MAX_FRAMES or min(width, height) < 16 or width % 16 or height % 16:
        raise HTTPException(400, "SeedVR2 long-video frame count or dimensions are invalid")
    if width * height > MAX_FRAME_PIXELS or max(width, height) > 1344 or prompt.strip():
        raise HTTPException(400, "SeedVR2 long video supports up to 768×1344 and a blank prompt")
    if not 0 <= seed <= 2**32 - 1:
        raise HTTPException(400, "Seed must be a 32-bit unsigned integer")
    if color_correction_method not in COLOR_CORRECTION_METHODS:
        raise HTTPException(400, f"color_correction_method must be one of {list(COLOR_CORRECTION_METHODS)}")
    async with job_lock:
        if active_task is not None and not active_task.done():
            raise HTTPException(409, "A SeedVR2 long-video job is already running")
        await asyncio.to_thread(_sweep_expired_jobs)
        job_id = uuid4().hex
        job = _jobs_root() / job_id
        job.mkdir(parents=True)
        try:
            await asyncio.to_thread(_store_upload, input_references, job / "input.mp4")
        except HTTPException:
            shutil.rmtree(job, ignore_errors=True)
            raise
        _status(job, "queued", 0)
        active_task = asyncio.create_task(
            asyncio.to_thread(
                _background,
                job,
                width,
                height,
                num_frames,
                loop_input,
                seed,
                color_correction_method,
                raw_request.app.state.seedvr2_long_port,
                raw_request.headers.get("authorization", ""),
            )
        )
    return {"id": job_id, "status": "queued", "max_frames": MAX_FRAMES}


@router.get("/v1/seedvr2/restore-long/{job_id}")
async def get_long_video(job_id: str) -> dict[str, str | int]:
    status = _job_dir(job_id) / "status.json"
    if not status.exists():
        raise HTTPException(404, "SeedVR2 long-video job not found")
    record = json.loads(status.read_text())
    if record["status"] in {"queued", "running"} and record.get("pid") != os.getpid():
        # Only this process runs jobs, so another owner means a restart lost it.
        record["status"], record["error"] = "failed", "SeedVR2 long-video job was interrupted by a server restart"
    return record


@router.get("/v1/seedvr2/restore-long/{job_id}/content")
async def download_long_video(job_id: str) -> FileResponse:
    job = _job_dir(job_id)
    if not (job / "result.json").exists():
        raise HTTPException(404, "SeedVR2 long-video output is not ready")
    return FileResponse(job / "output.mp4", media_type="video/mp4", filename=f"{job_id}.mp4")


@router.delete("/v1/seedvr2/restore-long/{job_id}", status_code=202)
async def cancel_long_video(job_id: str) -> dict[str, str]:
    """Ask the running job to stop; it settles at the next window boundary."""
    job = _job_dir(job_id)
    status = job / "status.json"
    if not status.exists():
        raise HTTPException(404, "SeedVR2 long-video job not found")
    record = json.loads(status.read_text())
    if record["status"] not in {"queued", "running"}:
        raise HTTPException(409, f"SeedVR2 long-video job already {record['status']}")
    (job / "cancel").touch()
    return {"id": job_id, "status": "cancelling"}
