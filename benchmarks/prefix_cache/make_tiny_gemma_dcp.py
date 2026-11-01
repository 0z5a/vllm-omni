"""Create a small random Gemma checkpoint for DCP runtime verification."""

import json
import sys
from pathlib import Path

import torch
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GemmaConfig, GemmaForCausalLM, PreTrainedTokenizerFast

source = Path(sys.argv[1])
target = Path(sys.argv[2])
target.mkdir(parents=True, exist_ok=True)
config_data = json.loads((source / "config.json").read_text())
config_data.update(
    hidden_size=256,
    intermediate_size=512,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=1,
    head_dim=64,
    vocab_size=256,
    max_position_embeddings=2048,
    bos_token_id=2,
    eos_token_id=3,
    pad_token_id=0,
)
config = GemmaConfig(**config_data)
torch.manual_seed(42)
GemmaForCausalLM(config).save_pretrained(target, safe_serialization=True)
vocab = {"[PAD]": 0, "[UNK]": 1, "[BOS]": 2, "[EOS]": 3}
for word in "The garden contains flowers birds and a quiet path beside small pond".split():
    vocab[word] = len(vocab)
tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
PreTrainedTokenizerFast(
    tokenizer_object=tokenizer,
    unk_token="[UNK]",
    bos_token="[BOS]",
    eos_token="[EOS]",
    pad_token="[PAD]",
).save_pretrained(target)
print(f"created {target}")
