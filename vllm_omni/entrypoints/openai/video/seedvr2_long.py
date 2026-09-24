# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""One asynchronous SeedVR2 request, bounded model windows, one MP4 writer."""

import asyncio
import hashlib
import io
import itertools
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import av
import imageio_ffmpeg
import numpy as np
import requests
from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse

from vllm_omni.inputs.data import COLOR_CORRECTION_METHODS, DEFAULT_COLOR_CORRECTION_METHOD

router = APIRouter()
MAX_FRAMES = 7200
MAX_FRAME_PIXELS = 768 * 1344
FPS = 24
WINDOW, OVERLAP = 12, 4
JOBS = Path(os.environ.get("SEEDVR2_LONG_OUTPUT_DIR", tempfile.gettempdir())) / "seedvr2-long"
active_task: asyncio.Task[None] | None = None
job_lock = asyncio.Lock()


def _status(job: Path, stage: str, frames: int, error: str = "") -> None:
    pending = job / "status.json.tmp"
    pending.write_text(json.dumps({"status": stage, "frames": frames, "error": error}) + "\n")
    pending.replace(job / "status.json")


def _frames(source: Path, width: int, height: int, loop: bool) -> Iterator[av.VideoFrame]:
    while True:
        count = 0
        with av.open(str(source)) as container:
            video = container.streams.video[0]
            if video.average_rate != Fraction(FPS):
                raise ValueError("SeedVR2 long video requires 24 FPS input")
            for frame in container.decode(video=0):
                if frame.pts is None or frame.time_base is None or frame.pts * frame.time_base != Fraction(count, FPS):
                    raise ValueError("SeedVR2 long video requires constant 24 FPS timestamps starting at zero")
                count += 1
                yield frame.reformat(width=width, height=height, format="yuv420p", interpolation="BICUBIC")
        if count == 0:
            raise ValueError("Input video has no frames")
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
    response.raise_for_status()
    with av.open(io.BytesIO(response.content)) as container:
        video = container.streams.video[0]
        decoded = list(container.decode(video=0))
        if (video.width, video.height, video.average_rate, len(decoded)) != (width, height, Fraction(FPS), len(frames)):
            raise ValueError("SeedVR2 window returned the wrong video geometry or frame count")
        if [frame.pts * frame.time_base for frame in decoded] != [Fraction(index, FPS) for index in range(len(frames))]:
            raise ValueError("SeedVR2 window returned invalid timestamps")
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
        raise ValueError("Input video has fewer frames than requested; set loop_input=true to repeat it")
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
                raise ValueError("Input video has fewer frames than requested; set loop_input=true to repeat it")
            batch = batch[-OVERLAP:] + next_frames
            start += len(restored) - OVERLAP
        for packet in video.encode():
            container.mux(packet)
    if written != target:
        raise ValueError(f"SeedVR2 restored {written} of {target} requested frames")

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
                raise ValueError("SeedVR2 final MP4 has invalid timestamps")
        if (video.width, video.height, video.average_rate, decoded) != (width, height, Fraction(FPS), target):
            raise ValueError("SeedVR2 final MP4 failed frame or geometry validation")
        if has_audio and not container.streams.audio:
            raise ValueError("SeedVR2 final MP4 lost its audio track")
    if has_audio:
        with av.open(str(output)) as container:
            audio = container.streams.audio[0]
            samples = sum(frame.samples for frame in container.decode(audio=0))
            if samples < (target / FPS - 1) * audio.rate:
                raise ValueError("SeedVR2 final MP4 audio ends before the video")
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
        current = json.loads((job / "status.json").read_text())
        _status(job, "failed", current["frames"], str(error))


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
        job_id = uuid4().hex
        job = JOBS / job_id
        job.mkdir(parents=True)
        with (job / "input.mp4").open("wb") as destination:
            await asyncio.to_thread(shutil.copyfileobj, input_references.file, destination)
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
    status = JOBS / job_id / "status.json"
    if len(job_id) != 32 or not status.exists():
        raise HTTPException(404, "SeedVR2 long-video job not found")
    return json.loads(status.read_text())


@router.get("/v1/seedvr2/restore-long/{job_id}/content")
async def download_long_video(job_id: str) -> FileResponse:
    job = JOBS / job_id
    if len(job_id) != 32 or not (job / "result.json").exists():
        raise HTTPException(404, "SeedVR2 long-video output is not ready")
    return FileResponse(job / "output.mp4", media_type="video/mp4", filename=f"{job_id}.mp4")
