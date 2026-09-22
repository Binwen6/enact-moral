from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Any

import torch

from config import load_model_specs
from io_utils import read_jsonl, write_json
from modeling import input_device, load_model
from readouts import expected_rating, label_token_ids


CELLS = ("direct", "mediated_controllable", "mediated_uncontrollable")
MEDIATOR_FIELDS = {
    "mediated_controllable": "intermediate_event_controllable",
    "mediated_uncontrollable": "intermediate_event_uncontrollable",
}
CAUSAL_FIELDS = {
    "mediated_controllable": "mediated_controllable_sentence",
    "mediated_uncontrollable": "mediated_uncontrollable_sentence",
}


def fill(template: str, record: dict[str, Any]) -> str:
    return template.format(
        AGENT=record["cast"]["agent"]["name"],
        PATIENT=record["cast"]["patient"]["name"],
    )


def causation_prompt(record: dict[str, Any], cell: str) -> str:
    agent = record["cast"]["agent"]["name"]
    question = (
        f"Please rate how strongly {agent}'s action caused what happened, "
        "on a scale from 1 (not at all) to 7 (entirely). Rating:"
    )
    return f"{record['cells'][cell]['scenario']} {question}"


def token_span(tokenizer: Any, prompt: str, character_start: int, character_end: int) -> list[int]:
    encoded = tokenizer(prompt, return_offsets_mapping=True)
    span = [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if right > character_start and left < character_end and right > left
    ]
    if not span:
        raise ValueError("Character span covers no tokens")
    return span


def prepared_spans(
    tokenizer: Any,
    record: dict[str, Any],
    cell: str,
    generator: random.Random,
    random_placements: int,
) -> tuple[str, list[int], list[list[int]]]:
    prompt = causation_prompt(record, cell)
    scenario = record["cells"][cell]["scenario"]
    causal_text = fill(record["fields_raw"][CAUSAL_FIELDS[cell]], record)
    mediator_text = fill(record["fields_raw"][MEDIATOR_FIELDS[cell]], record)
    causal_start = scenario.find(causal_text)
    if causal_start < 0:
        raise ValueError(f"Causal sentence is absent: {record['unit_id']}/{cell}")
    mediator_offset = causal_text.lower().find(mediator_text.lower())
    if mediator_offset < 0:
        raise ValueError(f"Mediator is absent from its causal sentence: {record['unit_id']}/{cell}")
    mediator_start = causal_start + mediator_offset
    mediator = token_span(tokenizer, prompt, mediator_start, mediator_start + len(mediator_text))
    prefix = token_span(tokenizer, prompt, 0, causal_start)
    width = len(mediator)
    starts = list(range(len(prefix) - width + 1))
    if len(starts) < random_placements:
        raise ValueError(f"Insufficient random-span placements: {record['unit_id']}/{cell}")
    selected_starts = generator.sample(starts, random_placements)
    random_spans = [prefix[start : start + width] for start in selected_starts]
    return prompt, mediator, random_spans


def rating(
    model: Any,
    tokenizer: Any,
    prompt: str,
    rating_ids: torch.Tensor,
    blocked_span: list[int] | None,
) -> float:
    batch = tokenizer(prompt, return_tensors="pt")
    device = input_device(model)
    input_ids = batch["input_ids"].to(device)
    length = input_ids.shape[1]
    positions = torch.arange(length, device=device).unsqueeze(0)
    attention_mask: torch.Tensor
    if blocked_span is None:
        attention_mask = batch["attention_mask"].to(device)
    else:
        allowed = torch.ones((length, length), dtype=torch.bool, device=device).tril()
        first_query = max(blocked_span) + 1
        allowed[first_query:, torch.tensor(blocked_span, device=device)] = False
        attention_mask = allowed.unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions,
            use_cache=False,
        ).logits[:, -1, :].float().cpu()
    return float(expected_rating(logits, rating_ids)[0])


def main() -> int:
    parser = argparse.ArgumentParser(description="Remove downstream attention to intermediate-event spans.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source", choices=("local", "hub"), required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--random-placements", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    if arguments.random_placements < 1:
        raise ValueError("random-placements must be positive")
    specs = load_model_specs()
    if arguments.model_id not in specs:
        raise KeyError(f"Unknown model: {arguments.model_id}")
    records = read_jsonl(arguments.data)
    if arguments.max_records is not None:
        records = records[: arguments.max_records]
    model, tokenizer, metadata = load_model(
        specs[arguments.model_id], arguments.model_source, arguments.model_root,
        arguments.revision, arguments.device_map,
    )
    rating_ids = label_token_ids(tokenizer, [str(value) for value in range(1, 8)])
    generator = random.Random(arguments.seed)
    rows: list[dict[str, Any]] = []
    for record in records:
        baselines = {
            cell: rating(model, tokenizer, causation_prompt(record, cell), rating_ids, None)
            for cell in CELLS
        }
        row: dict[str, Any] = {
            "record_id": record["unit_id"],
            "direct_baseline": baselines["direct"],
            "controllable_baseline": baselines["mediated_controllable"],
            "uncontrollable_baseline": baselines["mediated_uncontrollable"],
        }
        for cell, prefix in (
            ("mediated_controllable", "controllable"),
            ("mediated_uncontrollable", "uncontrollable"),
        ):
            prompt, mediator, random_spans = prepared_spans(
                tokenizer, record, cell, generator, arguments.random_placements,
            )
            mediator_score = rating(model, tokenizer, prompt, rating_ids, mediator)
            random_scores = [
                rating(model, tokenizer, prompt, rating_ids, span)
                for span in random_spans
            ]
            row[f"{prefix}_mediator_delta"] = mediator_score - baselines[cell]
            row[f"{prefix}_random_delta"] = sum(
                value - baselines[cell] for value in random_scores
            ) / len(random_scores)
            row[f"{prefix}_mediator_tokens"] = len(mediator)
        rows.append(row)
    summary_keys = [
        "direct_baseline",
        "controllable_baseline",
        "uncontrollable_baseline",
        "controllable_mediator_delta",
        "controllable_random_delta",
        "uncontrollable_mediator_delta",
        "uncontrollable_random_delta",
    ]
    result = {
        "model": metadata,
        "seed": arguments.seed,
        "random_placements": arguments.random_placements,
        "summary": {
            key: sum(float(row[key]) for row in rows) / len(rows)
            for key in summary_keys
        },
        "records": rows,
    }
    write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
