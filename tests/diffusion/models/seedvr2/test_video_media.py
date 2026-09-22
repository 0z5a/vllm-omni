# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from vllm_omni.diffusion.ipc import _pack_diffusion_media, _unpack_diffusion_media
from vllm_omni.diffusion.media import (
    DiffusionMediaOutput,
    VideoMediaOutput,
    VideoTensorEncoding,
    VideoTensorLayout,
    VideoTensorSpec,
    VideoValueRange,
    slice_diffusion_media_output,
)
from vllm_omni.diffusion.models.seedvr2.video import read_video
from vllm_omni.diffusion.postprocess.media import finalize_diffusion_media
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.mark.parametrize("samples", [320, 160000])
def test_fractional_fps_and_pcm_survive_media_ipc_and_finalization(samples):
    audio = torch.linspace(-0.5, 0.5, samples * 2).reshape(2, 1, samples)
    media = DiffusionMediaOutput(
        video=VideoMediaOutput(
            tensor=torch.zeros(2, 3, 5, 16, 16),
            spec=VideoTensorSpec(
                VideoTensorLayout.BCTHW, VideoTensorEncoding.NORMALIZED_FLOAT, VideoValueRange.ZERO_TO_ONE
            ),
        ),
        prepared_for_transport=True,
        fps=30000 / 1001,
        audio=audio,
        audio_sample_rate=16000,
    )
    restored = _unpack_diffusion_media(_pack_diffusion_media(media, d2h_stream=None))
    assert restored.fps == media.fps
    assert restored.audio_sample_rate == 16000
    torch.testing.assert_close(restored.audio, audio, rtol=0, atol=0)
    result = finalize_diffusion_media(restored, sampling_params=OmniDiffusionSamplingParams())
    assert result["metadata"]["video"]["fps"] == 30000 / 1001
    assert result["metadata"]["audio"]["sample_rate"] == 16000
    torch.testing.assert_close(result["payload"]["audio"], audio, rtol=0, atol=0)
    second = slice_diffusion_media_output(restored, 1, 2).to_cpu()
    assert second.video.tensor.shape[0] == second.audio.shape[0] == 1
    torch.testing.assert_close(second.audio, audio[1:2], rtol=0, atol=0)


def test_read_real_video_preserves_frame_order_and_timestamps(tmp_path):
    path = tmp_path / "six-frames.mkv"
    with av.open(str(path), "w") as container:
        stream = container.add_stream("ffv1", rate=Fraction(30000, 1001))
        stream.width, stream.height, stream.pix_fmt = 32, 16, "bgr0"
        for index in range(6):
            frame = av.VideoFrame.from_ndarray(np.full((16, 32, 3), index * 30, np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    frames, source = read_video(path)
    assert frames.shape == (6, 3, 16, 32)
    assert source.fps == 30000 / 1001
    assert len(source.pts) == 6
    assert all(b > a for a, b in zip(source.pts, source.pts[1:]))
    assert source.audio is None and source.audio_sample_rate is None
    for index in range(6):
        torch.testing.assert_close(frames[index], torch.full_like(frames[index], index * 30 / 255))


@pytest.mark.parametrize("audio_offset", [0, 800])
def test_read_video_keeps_pcm_audio(tmp_path, audio_offset):
    path = tmp_path / "with-audio.mkv"
    samples = (np.sin(np.arange(3200) * 2 * np.pi * 440 / 16000) * 12000).astype(np.int16)[None]
    with av.open(str(path), "w") as container:
        video = container.add_stream("ffv1", rate=25)
        video.width, video.height, video.pix_fmt = 32, 16, "bgr0"
        audio = container.add_stream("pcm_s16le", rate=16000)
        audio.layout = "mono"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 32, 3), np.uint8), format="rgb24")
            for packet in video.encode(frame):
                container.mux(packet)
        for packet in video.encode():
            container.mux(packet)
        frame = av.AudioFrame.from_ndarray(samples, format="s16", layout="mono")
        frame.sample_rate, frame.pts, frame.time_base = 16000, audio_offset, Fraction(1, 16000)
        for packet in audio.encode(frame):
            container.mux(packet)
        for packet in audio.encode():
            container.mux(packet)
    frames, source = read_video(path)
    assert frames.shape[0] == 5
    assert source.audio_sample_rate == 16000
    expected = np.zeros_like(samples, dtype=np.float32)
    expected[:, audio_offset:] = samples[:, : 3200 - audio_offset].astype(np.float32) / 32768
    torch.testing.assert_close(source.audio[0], torch.from_numpy(expected), rtol=0, atol=0)


def test_corrupt_video_is_a_client_error(tmp_path):
    from vllm_omni.errors import OmniClientError

    path = tmp_path / "broken.mkv"
    path.write_bytes(b"not a video")
    with pytest.raises(OmniClientError, match="Cannot decode"):
        read_video(path)
