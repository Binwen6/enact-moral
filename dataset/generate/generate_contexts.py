from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


def read_graphs(path: Path) -> list[dict[str, Any]]:
    graphs: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                graph = json.loads(line)
                if "graph_template_id" not in graph or "settings" not in graph:
                    raise ValueError("Every graph requires graph_template_id and settings")
                graphs.append(graph)
    if not graphs:
        raise ValueError(f"No graphs in {path}")
    return graphs


def request_contexts(graph: dict[str, Any], base_url: str, api_key: str, model: str) -> dict[str, Any]:
    system = (
        "Return one JSON object with a settings mapping. Preserve every graph role and binding. "
        "Each setting value must contain one neutral context sentence of 8 to 24 words. "
        "Do not add causal connectives, outcomes, moral labels, or patient descriptions."
    )
    response = requests.post(
        base_url.rstrip("/") + "/chat/completions",
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json={
            "model": model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(graph, ensure_ascii=False)},
            ],
        },
        timeout=180,
    )
    response.raise_for_status()
    content = response.json()["choices"][0]["message"]["content"]
    value = json.loads(content)
    if not isinstance(value, dict):
        raise TypeError("Generation response must be a JSON object")
    return value


def validate_contexts(graph: dict[str, Any], generated: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    settings = generated.get("settings")
    if not isinstance(settings, dict):
        return ["missing settings mapping"]
    if set(settings) != set(graph["settings"]):
        errors.append("setting identifiers differ from the frozen graph")
    banned = ("because", "caused", "therefore", "moral", "wrong", "harm", "benefit")
    for setting_id, value in settings.items():
        context = value.get("context") if isinstance(value, dict) else None
        if not isinstance(context, str):
            errors.append(f"{setting_id}: missing context")
            continue
        count = len(context.split())
        if count < 8 or count > 24:
            errors.append(f"{setting_id}: context has {count} words")
        lowered = context.lower()
        hits = [word for word in banned if word in lowered]
        if hits:
            errors.append(f"{setting_id}: banned terms {hits}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate surface contexts for frozen MORALTRACE graphs.")
    parser.add_argument("--graphs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    base_url = os.environ["OPENAI_BASE_URL"]
    api_key = os.environ["OPENAI_API_KEY"]
    model = os.environ["DATAGEN_MODEL"]
    if arguments.output.exists():
        raise FileExistsError(arguments.output)

    with arguments.output.open("x", encoding="utf-8") as handle:
        for graph in read_graphs(arguments.graphs):
            generated: dict[str, Any] | None = None
            errors: list[str] = []
            for attempt in range(1, 4):
                generated = request_contexts(graph, base_url, api_key, model)
                errors = validate_contexts(graph, generated)
                if not errors:
                    break
                graph = {**graph, "repair_feedback": errors, "repair_attempt": attempt}
            if errors or generated is None:
                raise ValueError(f"Generation failed for {graph['graph_template_id']}: {errors}")
            record = {
                "graph_template_id": graph["graph_template_id"],
                "settings": generated["settings"],
                "provenance": {
                    "generator_model": model,
                    "graph_frozen_before_surface_generation": True,
                },
            }
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

