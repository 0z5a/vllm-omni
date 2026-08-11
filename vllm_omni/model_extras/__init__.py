# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm_omni.model_extras.registry import (
    adapt_image_to_video_prompt,
    adapt_text_to_image_prompt,
    build_image_to_image_prompt,
    build_x_to_text_prompt,
    get_extra_body_params,
    get_extra_output_params,
    get_model_class_name,
    get_x_to_text_model_family,
    should_init_extra_args_for_non_diffusion_stages,
)

__all__ = [
    "adapt_image_to_video_prompt",
    "adapt_text_to_image_prompt",
    "build_image_to_image_prompt",
    "build_x_to_text_prompt",
    "get_extra_body_params",
    "get_extra_output_params",
    "get_model_class_name",
    "get_x_to_text_model_family",
    "should_init_extra_args_for_non_diffusion_stages",
]
