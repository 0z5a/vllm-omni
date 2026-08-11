# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
from PIL import Image

from vllm_omni.diffusion.utils.image_output import extract_images_from_outputs
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.diffusion, pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize(
    ("payload", "expected_pixel"),
    [
        (np.array([[[-1.0, 0.0, 1.0]]], dtype=np.float32), (0, 127, 255)),
        (np.array([[[0, 128, 255]]], dtype=np.uint8), (0, 128, 255)),
    ],
)
def test_extract_images_normalizes_numpy_payload(
    payload: np.ndarray,
    expected_pixel: tuple[int, int, int],
    tmp_path,
) -> None:
    output = OmniRequestOutput(images=[payload])

    [image] = extract_images_from_outputs(output)

    assert isinstance(image, Image.Image)
    assert image.mode == "RGB"
    assert image.size == (1, 1)
    assert image.getpixel((0, 0)) == expected_pixel

    output_path = tmp_path / "image.png"
    image.save(output_path)
    assert output_path.is_file()
