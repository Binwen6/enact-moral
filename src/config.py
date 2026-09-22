from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    hub_id: str
    local_name: str
    family: str
    parameters_billion: float
    dtype: str
    use_chat_template: bool
    disable_thinking: bool


def read_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a mapping in {path}")
    return value


def load_model_specs(path: Path | None = None) -> dict[str, ModelSpec]:
    config_path = path or PROJECT_ROOT / "configs" / "models.yaml"
    payload = read_yaml(config_path)
    models = payload.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError(f"No models declared in {config_path}")
    specs: dict[str, ModelSpec] = {}
    for model_id, raw in models.items():
        if not isinstance(raw, dict):
            raise TypeError(f"Invalid model entry: {model_id}")
        specs[model_id] = ModelSpec(model_id=model_id, **raw)
    return specs
