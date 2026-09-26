from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from config import load_model_specs
from interventions import PatchSpec, capture_states, run_with_patches
from io_utils import read_jsonl, write_json
from modeling import load_model, render_chat, token_positions_for_char_ends
from readouts import single_token_id


def cell_key(record: dict[str, Any]) -> tuple[str, str, str, str, str]:
    manipulations = record["manipulations"]
    return (
        record["graph_template_id"],
        record["semantic_setting"],
        manipulations["affected_role_realization"],
        manipulations["causal_connection"],
        manipulations["outcome_valence"],
    )


def opposite(value: str, pair: tuple[str, str]) -> str:
    if value == pair[0]:
        return pair[1]
    if value == pair[1]:
        return pair[0]
    raise ValueError(f"Unexpected factorial value: {value}")


def donor_indices(records: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
    lookup = {cell_key(record): index for index, record in enumerate(records)}
    relation_values = tuple(sorted({record["manipulations"]["causal_connection"] for record in records}))
    valence_values = tuple(sorted({record["manipulations"]["outcome_valence"] for record in records}))
    if len(relation_values) != 2 or len(valence_values) != 2:
        raise ValueError("Relational binding requires binary relation and valence factors")
    relation: list[int] = []
    valence: list[int] = []
    joint: list[int] = []
    for record in records:
        graph, setting, affected, source, outcome = cell_key(record)
        other_source = opposite(source, relation_values)
        other_outcome = opposite(outcome, valence_values)
        relation.append(lookup[(graph, setting, affected, other_source, outcome)])
        valence.append(lookup[(graph, setting, affected, source, other_outcome)])
        joint.append(lookup[(graph, setting, affected, other_source, other_outcome)])
    return {
        "relation": torch.tensor(relation, dtype=torch.long),
        "valence": torch.tensor(valence, dtype=torch.long),
        "joint": torch.tensor(joint, dtype=torch.long),
    }


def intention_prompt(record: dict[str, Any]) -> str:
    for query in record["behavioral_diagnostic_queries"]:
        if query["query_id"] == "inferred_intention":
            return query["prompt"]
    raise KeyError(f"Missing inferred_intention query: {record['record_id']}")


def segment_character_end(record: dict[str, Any], raw_prompt: str, segment_name: str) -> int:
    segment = record["segments_by_causal_clause_order"]["source_first"][segment_name]
    if raw_prompt.count(segment) != 1:
        raise ValueError(
            f"Expected one {segment_name} segment in the intention prompt: {record['record_id']}"
        )
    return raw_prompt.index(segment) + len(segment)


def rendered_segment_positions(
    tokenizer: Any,
    records: list[dict[str, Any]],
    raw_prompts: list[str],
    rendered_prompts: list[str],
    segment_name: str,
) -> torch.Tensor:
    character_ends: list[int] = []
    for record, raw_prompt, rendered_prompt in zip(records, raw_prompts, rendered_prompts, strict=True):
        raw_start = rendered_prompt.find(raw_prompt)
        if raw_start < 0:
            raise ValueError(f"Raw prompt is absent from rendered chat: {record['record_id']}")
        character_ends.append(raw_start + segment_character_end(record, raw_prompt, segment_name))
    return token_positions_for_char_ends(tokenizer, rendered_prompts, character_ends)


def select_complete_graphs(records: list[dict[str, Any]], split: str, setting: str, maximum: int) -> list[dict[str, Any]]:
    if maximum < 1:
        raise ValueError("maximum must be positive")
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        # Construction controls carry no factorial cell
        if record.get("record_type") == "core_factorial" and record.get("split") == split and record.get("semantic_setting") == setting:
            grouped[record["graph_template_id"]].append(record)
    selected: list[dict[str, Any]] = []
    for graph_id in sorted(grouped)[:maximum]:
        graph_records = grouped[graph_id]
        keys = {cell_key(record) for record in graph_records}
        if len(keys) != len(graph_records):
            raise ValueError(f"Duplicate factorial cell in {graph_id}")
        affected = {record["manipulations"]["affected_role_realization"] for record in graph_records}
        relations = {record["manipulations"]["causal_connection"] for record in graph_records}
        valences = {record["manipulations"]["outcome_valence"] for record in graph_records}
        expected = len(affected) * len(relations) * len(valences)
        if len(affected) != 2 or len(relations) != 2 or len(valences) != 2 or len(keys) != expected:
            raise ValueError(f"Incomplete 2x2x2 factorial graph: {graph_id}")
        selected.extend(graph_records)
    if not selected:
        raise ValueError("No complete graph was selected")
    return selected


def main() -> int:
    parser = argparse.ArgumentParser(description="Run causal-source and outcome-valence state replacement.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source", choices=("local", "hub"), required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--split", default="validation")
    parser.add_argument("--setting", default="normative_social")
    parser.add_argument("--max-graphs", type=int, default=1)
    parser.add_argument("--relation-layer", type=int, required=True)
    parser.add_argument("--valence-layer", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    specs = load_model_specs()
    if arguments.model_id not in specs:
        raise KeyError(f"Unknown model: {arguments.model_id}")
    records = select_complete_graphs(read_jsonl(arguments.data), arguments.split, arguments.setting, arguments.max_graphs)
    donors = donor_indices(records)
    model, tokenizer, metadata = load_model(
        specs[arguments.model_id], arguments.model_source, arguments.model_root,
        arguments.revision, arguments.device_map,
    )
    raw_prompts = [intention_prompt(record) for record in records]
    prompts = [render_chat(tokenizer, specs[arguments.model_id], prompt) for prompt in raw_prompts]
    relation_positions = rendered_segment_positions(
        tokenizer, records, raw_prompts, prompts, "causal_statement",
    )
    valence_positions = rendered_segment_positions(
        tokenizer, records, raw_prompts, prompts, "outcome",
    )
    clean_logits, relation_states, _ = capture_states(
        model, tokenizer, prompts, [arguments.relation_layer], relation_positions,
    )
    _, valence_states, _ = capture_states(
        model, tokenizer, prompts, [arguments.valence_layer], valence_positions,
    )
    relation_values = relation_states[arguments.relation_layer].index_select(0, donors["relation"])
    valence_values = valence_states[arguments.valence_layer].index_select(0, donors["valence"])
    joint_relation_values = relation_states[arguments.relation_layer].index_select(0, donors["joint"])
    joint_valence_values = valence_states[arguments.valence_layer].index_select(0, donors["joint"])
    patched = {
        "relation": run_with_patches(model, tokenizer, prompts, [
            PatchSpec(arguments.relation_layer, relation_positions, relation_values),
        ]),
        "valence": run_with_patches(model, tokenizer, prompts, [
            PatchSpec(arguments.valence_layer, valence_positions, valence_values),
        ]),
        "joint": run_with_patches(model, tokenizer, prompts, [
            PatchSpec(arguments.relation_layer, relation_positions, joint_relation_values),
            PatchSpec(arguments.valence_layer, valence_positions, joint_valence_values),
        ]),
    }
    yes = single_token_id(tokenizer, "Yes")
    no = single_token_id(tokenizer, "No")
    clean_scores = clean_logits[:, yes] - clean_logits[:, no]
    patch_scores = {name: logits[:, yes] - logits[:, no] for name, logits in patched.items()}
    result = {
        "model": metadata,
        "relation_layer": arguments.relation_layer,
        "valence_layer": arguments.valence_layer,
        "relation_position": "causal_statement_end",
        "valence_position": "outcome_end",
        "setting": arguments.setting,
        "records": [
            {
                "record_id": record["record_id"],
                "clean": float(clean_scores[index]),
                "relation": float(patch_scores["relation"][index]),
                "valence": float(patch_scores["valence"][index]),
                "joint": float(patch_scores["joint"][index]),
                "relation_donor": records[int(donors["relation"][index])]["record_id"],
                "valence_donor": records[int(donors["valence"][index])]["record_id"],
                "joint_donor": records[int(donors["joint"][index])]["record_id"],
            }
            for index, record in enumerate(records)
        ],
    }
    write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
