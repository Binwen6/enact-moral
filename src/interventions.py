from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

import torch

from modeling import encode, input_device, transformer_layers


@dataclass(frozen=True)
class PatchSpec:
    layer: int
    positions: torch.Tensor
    values: torch.Tensor


class HiddenStateCapture(AbstractContextManager["HiddenStateCapture"]):
    def __init__(self, model: Any, layers: list[int]) -> None:
        blocks = transformer_layers(model)
        if min(layers) < 1 or max(layers) > len(blocks):
            raise ValueError(f"Layers must lie in [1, {len(blocks)}]")
        self.values: dict[int, torch.Tensor] = {}
        self.positions: torch.Tensor | None = None
        self.handles = [blocks[layer - 1].register_forward_hook(self._hook(layer)) for layer in layers]

    def _hook(self, layer: int):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            if self.positions is None:
                raise RuntimeError("Capture positions were not supplied")
            hidden = output[0] if isinstance(output, tuple) else output
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            positions = self.positions.to(hidden.device)
            self.values[layer] = hidden[rows, positions].detach().float().cpu()
        return capture

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()


class HiddenStatePatch(AbstractContextManager["HiddenStatePatch"]):
    def __init__(self, model: Any, patches: list[PatchSpec]) -> None:
        grouped: dict[int, list[PatchSpec]] = {}
        for patch in patches:
            grouped.setdefault(patch.layer, []).append(patch)
        blocks = transformer_layers(model)
        invalid = sorted(layer for layer in grouped if layer < 1 or layer > len(blocks))
        if invalid:
            raise ValueError(f"Patch layers lie outside [1, {len(blocks)}]: {invalid}")
        self.handles = [
            blocks[layer - 1].register_forward_hook(self._hook(layer_patches))
            for layer, layer_patches in grouped.items()
        ]

    def _hook(self, patches: list[PatchSpec]):
        def replace(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            rows = torch.arange(hidden.shape[0], device=hidden.device)
            for patch in patches:
                positions = patch.positions.to(hidden.device)
                values = patch.values.to(device=hidden.device, dtype=hidden.dtype)
                hidden[rows, positions] = values
            return output
        return replace

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()


class LastTokenWrite(AbstractContextManager["LastTokenWrite"]):
    def __init__(self, model: Any, layer: int) -> None:
        blocks = transformer_layers(model)
        if layer < 1 or layer > len(blocks):
            raise ValueError(f"Layer must lie in [1, {len(blocks)}]")
        self.mode = "off"
        self.captured: torch.Tensor | None = None
        self.source: torch.Tensor | None = None
        self.basis: torch.Tensor | None = None
        self.handle = blocks[layer - 1].register_forward_hook(self._hook)

    def _hook(self, _module: Any, _inputs: Any, output: Any) -> Any:
        hidden = output[0] if isinstance(output, tuple) else output
        # Left padding puts the final prompt token at index -1
        if self.mode == "capture":
            self.captured = hidden[:, -1].detach().float()
            return output
        if self.mode != "write":
            return output
        if self.source is None:
            raise RuntimeError("Write mode requires source states")
        current = hidden[:, -1]
        delta = self.source.to(device=current.device, dtype=current.dtype) - current
        if self.basis is not None:
            basis = self.basis.to(device=current.device, dtype=current.dtype)
            delta = (delta @ basis) @ basis.T
        patched = hidden.clone()
        patched[:, -1] = current + delta
        return (patched, *output[1:]) if isinstance(output, tuple) else patched

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.handle.remove()


def capture_states(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    layers: list[int],
    positions: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[int, torch.Tensor], dict[str, torch.Tensor]]:
    batch = encode(tokenizer, prompts, input_device(model))
    selected = positions
    if selected is None:
        selected = batch["attention_mask"].sum(dim=1).long() - 1
        left_padding = batch["attention_mask"].shape[1] - batch["attention_mask"].sum(dim=1)
        selected = selected + left_padding.long()
    capture = HiddenStateCapture(model, layers)
    capture.positions = selected
    with capture, torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, -1, :]
    return logits.float().cpu(), capture.values, batch


def run_with_patches(
    model: Any,
    tokenizer: Any,
    prompts: list[str],
    patches: list[PatchSpec],
) -> torch.Tensor:
    batch = encode(tokenizer, prompts, input_device(model))
    with HiddenStatePatch(model, patches), torch.inference_mode():
        logits = model(**batch, use_cache=False).logits[:, -1, :]
    return logits.float().cpu()


def orthonormal_basis(vectors: torch.Tensor, tolerance: float = 1e-8) -> torch.Tensor:
    if vectors.ndim != 2:
        raise ValueError("Expected [hidden, rank] vectors")
    basis, triangular = torch.linalg.qr(vectors.float(), mode="reduced")
    diagonal = torch.abs(torch.diagonal(triangular))
    rank = int((diagonal > tolerance).sum().item())
    if rank == 0:
        raise ValueError("Vectors have zero numerical rank")
    return basis[:, :rank]


def replace_projection(target: torch.Tensor, donor: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    delta = donor.float() - target.float()
    return target.float() + (delta @ basis) @ basis.T


def remove_projection(states: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    return states.float() - (states.float() @ basis) @ basis.T
