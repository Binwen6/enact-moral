from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from config import load_model_specs
from io_utils import read_jsonl, write_jsonl
from modeling import final_logits, load_model, render_chat
from readouts import expected_rating, label_token_ids, single_token_id


def query_payload(
    record: dict[str, Any],
    query_id: str,
    causal_clause_order: str,
    label_order: str,
) -> tuple[str, str | None]:
    if "queries" in record:
        query = record["queries"].get(query_id)
        if not isinstance(query, str):
            raise KeyError(f"Missing query {query_id} in {record['record_id']}")
        return f"{record['scene']}\n\nQuestion: {query}", None
    for query in record["behavioral_diagnostic_queries"]:
        if query["query_id"] == query_id and query.get("causal_clause_order", causal_clause_order) == causal_clause_order:
            prompt = query.get("prompt")
            if isinstance(prompt, str):
                return prompt, None
            variants = query.get("prompt_variants")
            if isinstance(variants, list):
                selected = [variant for variant in variants if variant.get("label_order") == label_order]
                if len(selected) != 1:
                    raise ValueError(
                        f"Expected one {label_order} variant for {query_id}: {record['record_id']}"
                    )
                return selected[0]["prompt"], selected[0]["factual_target_label"]
    raise KeyError(f"Missing query {query_id} in {record['record_id']}")


def score_rows(
    logits: torch.Tensor,
    tokenizer: Any,
    query_id: str,
    target_labels: list[str | None],
) -> list[float]:
    if query_id == "appraisal":
        rating_ids = label_token_ids(tokenizer, [str(value) for value in range(1, 8)])
        return expected_rating(logits, rating_ids).tolist()
    if query_id in {"affected_entity", "causal_source"}:
        if any(label not in {"A", "B"} for label in target_labels):
            raise ValueError(f"Factual query {query_id} requires A/B target labels")
        a_token = single_token_id(tokenizer, "A")
        b_token = single_token_id(tokenizer, "B")
        margins = logits[:, a_token] - logits[:, b_token]
        signs = torch.tensor([1.0 if label == "A" else -1.0 for label in target_labels])
        return (margins * signs).tolist()
    yes = single_token_id(tokenizer, "Yes")
    no = single_token_id(tokenizer, "No")
    return (logits[:, yes] - logits[:, no]).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Collect behavioral readouts for MoralTrace instances.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source", choices=("local", "hub"), required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--query", required=True)
    parser.add_argument("--causal-clause-order", choices=("source_first", "other_first"), default="source_first")
    parser.add_argument("--label-order", choices=("identity", "swapped"), default="identity")
    parser.add_argument("--split")
    parser.add_argument("--max-records", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    specs = load_model_specs()
    if arguments.model_id not in specs:
        raise KeyError(f"Unknown model: {arguments.model_id}")
    records = read_jsonl(arguments.data)
    if arguments.split:
        records = [record for record in records if record.get("split") == arguments.split]
    if arguments.max_records:
        records = records[: arguments.max_records]
    if not records:
        raise ValueError("No selected instances")

    model, tokenizer, metadata = load_model(
        specs[arguments.model_id], arguments.model_source, arguments.model_root,
        arguments.revision, arguments.device_map,
    )
    payloads = [
        query_payload(record, arguments.query, arguments.causal_clause_order, arguments.label_order)
        for record in records
    ]
    prompts = [render_chat(tokenizer, specs[arguments.model_id], prompt) for prompt, _ in payloads]
    target_labels = [target for _, target in payloads]
    values: list[float] = []
    for start in range(0, len(prompts), arguments.batch_size):
        logits = final_logits(model, tokenizer, prompts[start : start + arguments.batch_size])
        values.extend(
            score_rows(
                logits,
                tokenizer,
                arguments.query,
                target_labels[start : start + arguments.batch_size],
            )
        )
    rows = [
        {
            "record_id": record["record_id"],
            "model_id": arguments.model_id,
            "query_id": arguments.query,
            "causal_clause_order": arguments.causal_clause_order,
            "label_order": arguments.label_order,
            "target_label": target_labels[index],
            "score": score,
            "model_revision": metadata["revision"],
        }
        for index, (record, score) in enumerate(zip(records, values, strict=True))
    ]
    write_jsonl(arguments.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
