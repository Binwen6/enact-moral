from __future__ import annotations

import argparse
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from config import load_model_specs
from io_utils import read_jsonl, write_json
from modeling import encode, input_device, load_model, render_chat, transformer_layers
from readouts import single_token_id
from relational_binding import donor_indices, intention_prompt, select_complete_graphs


@dataclass(frozen=True)
class Component:
    layer: int
    kind: str
    index: int | None = None

    @property
    def key(self) -> str:
        suffix = "mlp" if self.kind == "mlp" else f"h{self.index}"
        return f"L{self.layer}.{suffix}"


def attention_layout(block: Any) -> tuple[Any, int, int]:
    attention = block.self_attn
    projection = attention.o_proj
    head_dim = int(attention.head_dim)
    input_width = int(projection.in_features)
    if input_width % head_dim:
        raise ValueError(f"Attention output width {input_width} is not divisible by head dimension {head_dim}")
    return projection, input_width // head_dim, head_dim


def parse_components(value: str, model: Any) -> list[Component]:
    blocks = transformer_layers(model)
    components: list[Component] = []
    for item in value.split(","):
        layer_text, component_text = item.split(".", maxsplit=1)
        if not layer_text.startswith("L"):
            raise ValueError(f"Component layer must use L<number>: {item}")
        layer = int(layer_text[1:])
        if layer < 1 or layer > len(blocks):
            raise ValueError(f"Layer {layer} lies outside [1, {len(blocks)}]")
        if component_text == "mlp":
            components.append(Component(layer, "mlp"))
            continue
        if not component_text.startswith("h"):
            raise ValueError(f"Component must be an MLP or attention head: {item}")
        head = int(component_text[1:])
        _, head_count, _ = attention_layout(blocks[layer - 1])
        if head < 0 or head >= head_count:
            raise ValueError(f"Head {head} lies outside [0, {head_count - 1}] at layer {layer}")
        components.append(Component(layer, "head", head))
    if len(set(components)) != len(components):
        raise ValueError("Duplicate components")
    return components


class ComponentCapture(AbstractContextManager["ComponentCapture"]):
    def __init__(self, model: Any, components: list[Component]) -> None:
        blocks = transformer_layers(model)
        self.values: dict[Component, torch.Tensor] = {}
        self.handles = []
        for component in components:
            block = blocks[component.layer - 1]
            if component.kind == "mlp":
                self.handles.append(block.mlp.register_forward_hook(self._mlp_hook(component)))
            else:
                projection, _, head_dim = attention_layout(block)
                self.handles.append(
                    projection.register_forward_pre_hook(self._head_hook(component, head_dim))
                )

    def _mlp_hook(self, component: Component):
        def capture(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            self.values[component] = hidden[:, -1].detach().float().cpu()
        return capture

    def _head_hook(self, component: Component, head_dim: int):
        def capture(_module: Any, inputs: tuple[Any, ...]) -> None:
            hidden = inputs[0]
            start = int(component.index) * head_dim
            self.values[component] = hidden[:, -1, start : start + head_dim].detach().float().cpu()
        return capture

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()


class ComponentPatch(AbstractContextManager["ComponentPatch"]):
    def __init__(self, model: Any, values: dict[Component, torch.Tensor]) -> None:
        blocks = transformer_layers(model)
        self.handles = []
        for component, component_values in values.items():
            block = blocks[component.layer - 1]
            if component.kind == "mlp":
                self.handles.append(block.mlp.register_forward_hook(self._mlp_hook(component_values)))
            else:
                projection, _, head_dim = attention_layout(block)
                self.handles.append(
                    projection.register_forward_pre_hook(
                        self._head_hook(component, component_values, head_dim)
                    )
                )

    def _mlp_hook(self, values: torch.Tensor):
        def replace(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            hidden[:, -1] = values.to(device=hidden.device, dtype=hidden.dtype)
            return output
        return replace

    def _head_hook(self, component: Component, values: torch.Tensor, head_dim: int):
        def replace(_module: Any, inputs: tuple[Any, ...]) -> tuple[Any, ...]:
            hidden = inputs[0].clone()
            start = int(component.index) * head_dim
            hidden[:, -1, start : start + head_dim] = values.to(
                device=hidden.device, dtype=hidden.dtype,
            )
            return (hidden, *inputs[1:])
        return replace

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()


def main() -> int:
    parser = argparse.ArgumentParser(description="Measure mediation by frozen attention and MLP components.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source", choices=("local", "hub"), required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--setting", default="normative_social")
    parser.add_argument("--max-graphs", type=int, default=1)
    parser.add_argument("--components", required=True, help="Comma-separated entries such as L19.h3,L22.mlp")
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    specs = load_model_specs()
    spec = specs[arguments.model_id]
    records = select_complete_graphs(
        read_jsonl(arguments.data), arguments.split, arguments.setting, arguments.max_graphs
    )
    donors = donor_indices(records)["joint"]
    model, tokenizer, metadata = load_model(
        spec, arguments.model_source, arguments.model_root, arguments.revision, arguments.device_map,
    )
    components = parse_components(arguments.components, model)
    prompts = [render_chat(tokenizer, spec, intention_prompt(record)) for record in records]
    batch = encode(tokenizer, prompts, input_device(model))
    capture = ComponentCapture(model, components)
    with capture, torch.inference_mode():
        clean_logits = model(**batch, use_cache=False).logits[:, -1, :].float().cpu()
    donor_values = {
        key: values.index_select(0, donors)
        for key, values in capture.values.items()
    }
    with ComponentPatch(model, donor_values), torch.inference_mode():
        patched_logits = model(**batch, use_cache=False).logits[:, -1, :].float().cpu()
    yes = single_token_id(tokenizer, "Yes")
    no = single_token_id(tokenizer, "No")
    clean = clean_logits[:, yes] - clean_logits[:, no]
    patched = patched_logits[:, yes] - patched_logits[:, no]
    desired = clean.index_select(0, donors)
    result = {
        "model": metadata,
        "components": [
            {"key": component.key, "layer": component.layer, "kind": component.kind, "index": component.index}
            for component in components
        ],
        "setting": arguments.setting,
        "mean_progress": float((torch.abs(desired - clean) - torch.abs(desired - patched)).mean()),
        "records": [
            {
                "record_id": record["record_id"],
                "clean": float(clean[index]),
                "patched": float(patched[index]),
                "desired": float(desired[index]),
            }
            for index, record in enumerate(records)
        ],
    }
    write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
