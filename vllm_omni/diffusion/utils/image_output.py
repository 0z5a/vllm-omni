# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from PIL import Image
from vllm.outputs import RequestOutput

from vllm_omni.outputs import OmniRequestOutput

ImageOutput = OmniRequestOutput | RequestOutput
ImageOutputs = ImageOutput | Sequence[ImageOutput] | None


def extract_images_from_outputs(outputs: ImageOutputs) -> list[Image.Image]:
    """Extract PIL images from known Omni output types.

    Supported inputs:
    - OmniRequestOutput / list[OmniRequestOutput]
    - RequestOutput / list[RequestOutput] (compatibility fallback)

    PIL images, NumPy arrays, and PyTorch tensors are normalized to PIL images.
    """
    for output in _iter_known_outputs(outputs):
        if isinstance(output, OmniRequestOutput):
            images = _coerce_images(output.images)
            if images:
                return images

        for payload in _iter_multimodal_image_payloads(output):
            images = _coerce_images(payload)
            if images:
                return images

    return []


def _iter_known_outputs(value: ImageOutputs) -> Iterable[ImageOutput]:
    if value is None:
        return
    if isinstance(value, OmniRequestOutput | RequestOutput):
        yield value
        return
    if isinstance(value, Sequence):
        for item in value:
            if isinstance(item, OmniRequestOutput | RequestOutput):
                yield item


def _iter_multimodal_image_payloads(output: OmniRequestOutput | RequestOutput) -> Iterable[Any]:
    if isinstance(output, OmniRequestOutput):
        if output.multimodal_output is not None:
            yield from _image_values_from_mapping_like(output.multimodal_output)
        if output.request_output is not None:
            yield from _iter_request_output_payloads(output.request_output)
    else:
        yield from _iter_request_output_payloads(output)


def _iter_request_output_payloads(request_output: RequestOutput) -> Iterable[Any]:
    images = getattr(request_output, "images", None)
    if images is not None:
        yield images
    mm = getattr(request_output, "multimodal_output", None)
    if mm is not None:
        yield from _image_values_from_mapping_like(mm)
    for completion in getattr(request_output, "outputs", []):
        mm = getattr(completion, "multimodal_output", None)
        if mm is not None:
            yield from _image_values_from_mapping_like(mm)


def _image_values_from_mapping_like(value: Any) -> Iterable[Any]:
    if hasattr(value, "to_dict") and callable(value.to_dict):
        value = value.to_dict()
    if not isinstance(value, Mapping):
        return
    for key in ("image", "images", "model_outputs"):
        if key in value:
            yield value[key]


def _coerce_images(payload: Any) -> list[Image.Image]:
    if payload is None:
        return []
    if isinstance(payload, Image.Image):
        return [payload]
    if isinstance(payload, np.ndarray):
        return _ndarray_to_images(payload)
    if isinstance(payload, torch.Tensor):
        return _tensor_to_images(payload)
    if isinstance(payload, list | tuple):
        images: list[Image.Image] = []
        for item in payload:
            images.extend(_coerce_images(item))
        return images
    return []


def _ndarray_to_images(array: np.ndarray) -> list[Image.Image]:
    if array.ndim == 4:
        images: list[Image.Image] = []
        for single in array:
            images.extend(_ndarray_to_images(single))
        return images
    if array.ndim != 3 or array.size == 0:
        return []

    image = array
    if image.shape[-1] not in (1, 3, 4):
        if image.shape[0] not in (1, 3, 4):
            return []
        image = np.moveaxis(image, 0, -1)

    if np.issubdtype(image.dtype, np.floating):
        image = image.astype(np.float32, copy=False)
        if image.min() < 0.0:
            image = image / 2 + 0.5
        image = np.clip(image, 0.0, 1.0) * 255
    elif np.issubdtype(image.dtype, np.integer) or np.issubdtype(image.dtype, np.bool_):
        image = np.clip(image, 0, 255)
    else:
        return []

    image = image.astype(np.uint8)
    if image.shape[-1] == 1:
        image = image[..., 0]
    return [Image.fromarray(np.ascontiguousarray(image))]


def _tensor_to_images(tensor: torch.Tensor) -> list[Image.Image]:
    img = tensor.detach().to("cpu", dtype=torch.float32)
    if img.ndim == 4:
        images: list[Image.Image] = []
        for single in img:
            images.extend(_tensor_to_images(single))
        return images
    if img.ndim != 3:
        return []

    if img.shape[0] in (1, 3, 4):
        img = img.permute(1, 2, 0)

    # Some diffusion models emit image tensors normalized in [-1, 1].
    if img.min().item() < 0.0:
        img = img / 2 + 0.5
    img = img.clamp(0, 1).mul(255).to(torch.uint8).contiguous().numpy()
    return [Image.fromarray(img)]
