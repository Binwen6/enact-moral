from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from config import ModelSpec


def load_model(
    spec: ModelSpec,
    source: str,
    model_root: Path | None,
    revision: str | None,
    device_map: str,
) -> tuple[Any, Any, dict[str, Any]]:
    if source == "local":
        if model_root is None:
            raise ValueError("--model-root is required with --model-source local")
        model_path: str | Path = model_root / spec.local_name
        if not (Path(model_path) / "config.json").is_file():
            raise FileNotFoundError(Path(model_path) / "config.json")
        local_only = True
    elif source == "hub":
        if not revision:
            raise ValueError("--revision is required with --model-source hub")
        model_path = spec.hub_id
        local_only = False
    else:
        raise ValueError(f"Unsupported model source: {source}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=local_only,
        trust_remote_code=False,
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError(f"Tokenizer has no pad or EOS token: {spec.model_id}")
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    dtype = getattr(torch, spec.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        revision=revision,
        local_files_only=local_only,
        trust_remote_code=False,
        dtype=dtype,
        low_cpu_mem_usage=True,
        device_map=device_map,
    )
    model.eval()
    metadata = {
        "model": asdict(spec),
        "source": source,
        "resolved_path": str(model_path),
        "revision": revision,
        "dtype": spec.dtype,
        "device_map": device_map,
    }
    return model, tokenizer, metadata


def transformer_layers(model: Any) -> Any:
    model_type = str(model.config.model_type)
    if model_type in {
        "gemma",
        "gemma2",
        "gemma3_text",
        "llama",
        "mistral",
        "mixtral",
        "olmo",
        "olmo2",
        "olmo3",
        "qwen2",
        "qwen3",
    }:
        return model.model.layers
    raise ValueError(f"Unsupported architecture: {model_type}")


def input_device(model: Any) -> torch.device:
    return model.get_input_embeddings().weight.device


def render_chat(tokenizer: Any, spec: ModelSpec, prompt: str) -> str:
    if not spec.use_chat_template:
        return prompt
    arguments: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
    }
    if spec.disable_thinking:
        arguments["enable_thinking"] = False
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        **arguments,
    )
    if not isinstance(rendered, str):
        raise TypeError("Chat template did not return text")
    return rendered


def encode(tokenizer: Any, prompts: list[str], device: torch.device) -> dict[str, torch.Tensor]:
    batch = tokenizer(prompts, return_tensors="pt", padding=True)
    attention = batch["attention_mask"]
    position_ids = attention.long().cumsum(dim=-1) - 1
    position_ids.masked_fill_(attention == 0, 0)
    batch["position_ids"] = position_ids
    return {name: value.to(device) for name, value in batch.items()}


def token_positions_for_char_ends(
    tokenizer: Any,
    prompts: list[str],
    character_ends: list[int],
) -> torch.Tensor:
    if len(prompts) != len(character_ends):
        raise ValueError("Prompts and character endpoints must have the same length")
    encoded_rows = [
        tokenizer(prompt, return_offsets_mapping=True)
        for prompt in prompts
    ]
    maximum_length = max(len(row["input_ids"]) for row in encoded_rows)
    positions: list[int] = []
    for prompt, character_end, encoded in zip(prompts, character_ends, encoded_rows, strict=True):
        if character_end <= 0 or character_end > len(prompt):
            raise ValueError(f"Character endpoint {character_end} lies outside a prompt of length {len(prompt)}")
        candidates = [
            index
            for index, (left, right) in enumerate(encoded["offset_mapping"])
            if right > left and right <= character_end
        ]
        if not candidates:
            raise ValueError(f"No token ends at or before character endpoint {character_end}")
        token_index = candidates[-1]
        token_end = int(encoded["offset_mapping"][token_index][1])
        gap = prompt[token_end:character_end]
        if gap.strip():
            raise ValueError(f"Character endpoint cuts through an unaligned token: {gap!r}")
        positions.append(token_index + maximum_length - len(encoded["input_ids"]))
    return torch.tensor(positions, dtype=torch.long)


def final_logits(model: Any, tokenizer: Any, prompts: list[str]) -> torch.Tensor:
    batch = encode(tokenizer, prompts, input_device(model))
    with torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, -1, :]
    return logits.float().cpu()
