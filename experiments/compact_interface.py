from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from config import load_model_specs
from interventions import PatchSpec, capture_states, orthonormal_basis, replace_projection, run_with_patches
from io_utils import read_jsonl, write_json
from modeling import load_model, render_chat
from readouts import single_token_id
from relational_binding import donor_indices, intention_prompt, select_complete_graphs


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit and evaluate a low-rank ENACT interface on disjoint graphs.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source", choices=("local", "hub"), required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--setting", default="normative_social")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--train-graphs", type=int, default=1)
    parser.add_argument("--evaluation-graphs", type=int, default=1)
    parser.add_argument("--basis-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    specs = load_model_specs()
    spec = specs[arguments.model_id]
    all_records = read_jsonl(arguments.data)
    train = select_complete_graphs(all_records, "train", arguments.setting, arguments.train_graphs)
    evaluation = select_complete_graphs(all_records, "validation", arguments.setting, arguments.evaluation_graphs)
    model, tokenizer, metadata = load_model(
        spec, arguments.model_source, arguments.model_root, arguments.revision, arguments.device_map,
    )

    train_prompts = [render_chat(tokenizer, spec, intention_prompt(record)) for record in train]
    _, train_states_by_layer, _ = capture_states(model, tokenizer, train_prompts, [arguments.layer])
    train_states = train_states_by_layer[arguments.layer]
    train_donors = donor_indices(train)["joint"]
    changes = train_states.index_select(0, train_donors) - train_states
    _, _, right = torch.linalg.svd(changes.float(), full_matrices=False)
    if arguments.rank < 1 or arguments.rank > right.shape[0]:
        raise ValueError(f"Rank must lie in [1, {right.shape[0]}]")
    basis = orthonormal_basis(right[: arguments.rank].T)
    if arguments.basis_output is not None:
        arguments.basis_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(arguments.basis_output, basis.numpy())

    prompts = [render_chat(tokenizer, spec, intention_prompt(record)) for record in evaluation]
    clean_logits, states_by_layer, _ = capture_states(model, tokenizer, prompts, [arguments.layer])
    states = states_by_layer[arguments.layer]
    donors = donor_indices(evaluation)["joint"]
    donor_states = states.index_select(0, donors)
    projected = replace_projection(states, donor_states, basis)
    positions = torch.full((len(evaluation),), -1, dtype=torch.long)
    interface_logits = run_with_patches(
        model, tokenizer, prompts, [PatchSpec(arguments.layer, positions, projected)]
    )
    full_logits = run_with_patches(
        model, tokenizer, prompts, [PatchSpec(arguments.layer, positions, donor_states)]
    )
    yes = single_token_id(tokenizer, "Yes")
    no = single_token_id(tokenizer, "No")
    scores = {
        "clean": clean_logits[:, yes] - clean_logits[:, no],
        "interface": interface_logits[:, yes] - interface_logits[:, no],
        "full_state": full_logits[:, yes] - full_logits[:, no],
    }
    desired = scores["clean"].index_select(0, donors)
    baseline_distance = torch.abs(desired - scores["clean"])
    result = {
        "model": metadata,
        "setting": arguments.setting,
        "layer": arguments.layer,
        "rank": arguments.rank,
        "basis_file": arguments.basis_output.name if arguments.basis_output is not None else None,
        "train_graphs": arguments.train_graphs,
        "evaluation_graphs": arguments.evaluation_graphs,
        "mean_interface_progress": float((baseline_distance - torch.abs(desired - scores["interface"])).mean()),
        "mean_full_state_progress": float((baseline_distance - torch.abs(desired - scores["full_state"])).mean()),
        "records": [
            {
                "record_id": record["record_id"],
                "clean": float(scores["clean"][index]),
                "interface": float(scores["interface"][index]),
                "full_state": float(scores["full_state"][index]),
                "desired": float(desired[index]),
            }
            for index, record in enumerate(evaluation)
        ],
    }
    write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
