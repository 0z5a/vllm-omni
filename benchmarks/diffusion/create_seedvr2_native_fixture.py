# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Create the five-frame lossless HTTP fixture from the pinned Eyes clip."""

import argparse
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("source", type=Path)
parser.add_argument("output", type=Path)
args = parser.parse_args()
source, fixture = args.source, args.output
with av.open(str(source)) as container:
    frames = []
    for frame in container.decode(video=0):
        frames.append(frame.to_ndarray(format="rgb24"))
        if len(frames) == 5:
            break
with av.open(str(fixture), "w") as container:
    video = container.add_stream("ffv1", rate=25)
    video.width, video.height, video.pix_fmt = 212, 120, "bgr0"
    audio = container.add_stream("pcm_s16le", rate=16000)
    audio.layout = "mono"
    for pixels in frames:
        for packet in video.encode(av.VideoFrame.from_ndarray(pixels, format="rgb24")):
            container.mux(packet)
    for packet in video.encode():
        container.mux(packet)
    pcm = (np.sin(np.arange(3200) * 2 * np.pi * 440 / 16000) * 12000).astype(np.int16)[None]
    frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    frame.sample_rate, frame.pts, frame.time_base = 16000, 0, Fraction(1, 16000)
    for packet in audio.encode(frame):
        container.mux(packet)
    for packet in audio.encode():
        container.mux(packet)
