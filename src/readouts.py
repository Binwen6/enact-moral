from __future__ import annotations

from typing import Any, Iterable

import torch


def single_token_id(tokenizer: Any, label: str) -> int:
    candidates = (label, f" {label}", f"\n{label}")
    matches = [
        int(ids[0])
        for text in candidates
        if len(ids := tokenizer(text, add_special_tokens=False)["input_ids"]) == 1
    ]
    if not matches:
        raise ValueError(f"No single-token representation for {label!r}")
    return matches[0]


def label_token_ids(tokenizer: Any, labels: Iterable[str]) -> torch.Tensor:
    return torch.tensor([single_token_id(tokenizer, label) for label in labels], dtype=torch.long)


def binary_margin(logits: torch.Tensor, positive: int, negative: int) -> torch.Tensor:
    return logits[..., positive] - logits[..., negative]


def expected_rating(logits: torch.Tensor, rating_ids: torch.Tensor) -> torch.Tensor:
    selected = logits.index_select(-1, rating_ids.to(logits.device))
    probabilities = torch.softmax(selected.float(), dim=-1)
    values = torch.arange(1, len(rating_ids) + 1, device=logits.device, dtype=torch.float32)
    return probabilities @ values
