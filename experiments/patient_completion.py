from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from config import load_model_specs
from interventions import PatchSpec, capture_states, orthonormal_basis, remove_projection, run_with_patches
from io_utils import read_jsonl, write_json
from modeling import load_model, render_chat
from readouts import expected_rating, label_token_ids


UNITS = ("A", "B", "C", "D")


def flatten_prompts(records: list[dict[str, Any]], tokenizer: Any, model_spec: Any) -> tuple[list[str], list[tuple[int, str]]]:
    prompts: list[str] = []
    keys: list[tuple[int, str]] = []
    for record_index, record in enumerate(records):
        if set(record["units"]) != set(UNITS):
            raise ValueError(f"Incomplete quartet: {record['quartet_id']}")
        for unit in UNITS:
            prompts.append(render_chat(tokenizer, model_spec, record["units"][unit]["full_prompt"]))
            keys.append((record_index, unit))
    return prompts, keys


def row_indices(keys: list[tuple[int, str]], unit: str) -> torch.Tensor:
    return torch.tensor([index for index, (_, name) in enumerate(keys) if name == unit], dtype=torch.long)


def ratings(logits: torch.Tensor, rating_ids: torch.Tensor) -> list[float]:
    return expected_rating(logits, rating_ids).tolist()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run implicit moral-patient subspace interventions.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source", choices=("local", "hub"), required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--max-quartets", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    specs = load_model_specs()
    if arguments.model_id not in specs:
        raise KeyError(f"Unknown model: {arguments.model_id}")
    records = read_jsonl(arguments.data)
    if arguments.max_quartets:
        records = records[: arguments.max_quartets]
    model, tokenizer, metadata = load_model(
        specs[arguments.model_id], arguments.model_source, arguments.model_root,
        arguments.revision, arguments.device_map,
    )
    prompts, keys = flatten_prompts(records, tokenizer, specs[arguments.model_id])
    clean_logits, states_by_layer, encoded = capture_states(model, tokenizer, prompts, [arguments.layer])
    states = states_by_layer[arguments.layer]
    indices = {unit: row_indices(keys, unit) for unit in UNITS}
    explicit = (states[indices["A"]] - states[indices["C"]]).mean(dim=0)
    implicit = (states[indices["B"]] - states[indices["D"]]).mean(dim=0)
    basis_1d = orthonormal_basis(implicit[:, None])
    basis_2d = orthonormal_basis(torch.stack((explicit, implicit), dim=1))
    generator = torch.Generator().manual_seed(arguments.seed)
    random_vectors = torch.randn(states.shape[1], 2, generator=generator)
    random_basis = orthonormal_basis(random_vectors)
    final_positions = torch.full((len(records),), -1, dtype=torch.long)

    b_states = states[indices["B"]]
    d_states = states[indices["D"]]
    removal_1d = remove_projection(b_states, basis_1d)
    removal_2d = remove_projection(b_states, basis_2d)
    removal_random = remove_projection(b_states, random_basis)
    addition_1d = d_states + implicit
    addition_2d = d_states + explicit + implicit
    b_prompts = [prompts[index] for index in indices["B"].tolist()]
    d_prompts = [prompts[index] for index in indices["D"].tolist()]

    conditions = {
        "remove_1d": run_with_patches(model, tokenizer, b_prompts, [PatchSpec(arguments.layer, final_positions, removal_1d)]),
        "remove_2d": run_with_patches(model, tokenizer, b_prompts, [PatchSpec(arguments.layer, final_positions, removal_2d)]),
        "remove_random_2d": run_with_patches(model, tokenizer, b_prompts, [PatchSpec(arguments.layer, final_positions, removal_random)]),
        "add_1d": run_with_patches(model, tokenizer, d_prompts, [PatchSpec(arguments.layer, final_positions, addition_1d)]),
        "add_2d": run_with_patches(model, tokenizer, d_prompts, [PatchSpec(arguments.layer, final_positions, addition_2d)]),
    }
    rating_ids = label_token_ids(tokenizer, [str(value) for value in range(1, 8)])
    clean_b = clean_logits.index_select(0, indices["B"])
    clean_d = clean_logits.index_select(0, indices["D"])
    result = {
        "model": metadata,
        "layer": arguments.layer,
        "quartets": len(records),
        "direction_cosine": float(torch.nn.functional.cosine_similarity(explicit, implicit, dim=0)),
        "records": [
            {
                "quartet_id": record["quartet_id"],
                "clean_implicit": ratings(clean_b, rating_ids)[index],
                "clean_no_patient": ratings(clean_d, rating_ids)[index],
                **{name: ratings(logits, rating_ids)[index] for name, logits in conditions.items()},
            }
            for index, record in enumerate(records)
        ],
    }
    write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
