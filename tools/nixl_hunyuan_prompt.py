# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Use the checkpoint's DiT template for the validation AR input."""

from transformers import AutoTokenizer, GenerationConfig

from vllm_omni.diffusion.models.hunyuan_image3.hunyuan_image3_tokenizer import TokenizerWrapper


def load_prompt_builder(model: str) -> tuple[TokenizerWrapper, str]:
    tokenizer = TokenizerWrapper(AutoTokenizer.from_pretrained(model, trust_remote_code=True))
    config = GenerationConfig.from_pretrained(model).to_dict()
    return tokenizer, config.get("sequence_template", "pretrain")


def build_ar_tokens(tokenizer: TokenizerWrapper, text: str, sequence_template: str) -> list[int]:
    result = tokenizer.apply_chat_template(
        batch_prompt=[text],
        mode="gen_text",
        bot_task="think",
        sequence_template=sequence_template,
    )
    return result["output"].tokens[0].tolist()
