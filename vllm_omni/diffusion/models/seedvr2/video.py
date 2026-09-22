# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Decode restoration inputs before scheduler admission."""

from collections.abc import Iterator
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
import torch

from vllm_omni.errors import OmniClientError


@dataclass(frozen=True)
class SourceVideo:
    fps: float
    pts: tuple[int, ...]
    time_base: Fraction
    audio: torch.Tensor | None
    audio_sample_rate: int | None


def read_video(path: str | Path) -> tuple[torch.Tensor, SourceVideo]:
    try:
        return _read_video(path)
    except av.FFmpegError as exc:
        raise OmniClientError(f"Cannot decode SeedVR2 input video: {exc.strerror}") from exc


def _read_video(path: str | Path) -> tuple[torch.Tensor, SourceVideo]:
    with av.open(str(path)) as container:
        if not container.streams.video:
            raise OmniClientError("SeedVR2 input contains no video stream")
        stream = container.streams.video[0]
        rate = stream.average_rate
        if rate is None or rate <= 0:
            raise OmniClientError("SeedVR2 input must declare a positive frame rate")
        frames = []
        pts = []
        for frame in container.decode(stream):
            if frame.pts is None:
                raise OmniClientError("SeedVR2 input video is missing frame timestamps")
            frames.append(torch.from_numpy(frame.to_ndarray(format="rgb24")))
            pts.append(frame.pts)
        if not frames:
            raise OmniClientError("SeedVR2 input video contains no decoded frames")
        time_base = Fraction(stream.time_base)
        if any(b <= a for a, b in zip(pts, pts[1:])):
            raise OmniClientError("SeedVR2 input timestamps must increase")
        if any(
            abs((pts[index] - pts[0]) * time_base - Fraction(index, 1) / rate) > time_base for index in range(len(pts))
        ):
            raise OmniClientError("SeedVR2 currently supports constant-frame-rate video only")
    audio = None
    sample_rate = None
    with av.open(str(path)) as container:
        if container.streams.audio:
            stream = container.streams.audio[0]
            sample_rate = stream.codec_context.sample_rate
            resampler = av.AudioResampler(format="fltp", layout=stream.codec_context.layout, rate=sample_rate)
            channels = len(stream.codec_context.layout.channels)
            if channels not in (1, 2):
                raise OmniClientError("SeedVR2 audio preservation currently supports mono or stereo")
            sample_count = round(len(frames) / rate * sample_rate)
            waveform = np.zeros((channels, sample_count), dtype=np.float32)

            def resampled_frames() -> Iterator[av.AudioFrame]:
                for frame in container.decode(stream):
                    yield from resampler.resample(frame)
                yield from resampler.resample(None)

            for frame in resampled_frames():
                if frame.pts is None:
                    raise OmniClientError("SeedVR2 audio is missing timestamps")
                offset = round((frame.pts * frame.time_base - pts[0] * time_base) * sample_rate)
                chunk = frame.to_ndarray()
                start, stop = max(0, offset), min(sample_count, offset + chunk.shape[1])
                if stop > start:
                    waveform[:, start:stop] = chunk[:, start - offset : stop - offset]
            audio = torch.from_numpy(waveform).unsqueeze(0)
    return torch.stack(frames).permute(0, 3, 1, 2).float() / 255, SourceVideo(
        float(rate),
        tuple(pts),
        time_base,
        audio,
        sample_rate,
    )
