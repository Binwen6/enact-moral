from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch

from config import load_model_specs
from interventions import PatchSpec, capture_states, orthonormal_basis, replace_projection, run_with_patches
from io_utils import read_jsonl, write_json
from modeling import load_model, render_chat
from readouts import expected_rating, label_token_ids, single_token_id


def prompt_pair(record: dict[str, Any]) -> tuple[str, str]:
    target = record.get("target_prompt")
    donor = record.get("donor_prompt")
    if not isinstance(target, str) or not target.strip():
        raise ValueError(f"Missing target_prompt: {record.get('record_id')}")
    if not isinstance(donor, str) or not donor.strip():
        raise ValueError(f"Missing donor_prompt: {record.get('record_id')}")
    return target, donor


def readout_scores(logits: torch.Tensor, tokenizer: Any, readout: str) -> torch.Tensor:
    if readout == "yes_no":
        yes = single_token_id(tokenizer, "Yes")
        no = single_token_id(tokenizer, "No")
        return logits[:, yes] - logits[:, no]
    if readout == "rating_1_7":
        rating_ids = label_token_ids(tokenizer, [str(value) for value in range(1, 8)])
        return expected_rating(logits, rating_ids)
    raise ValueError(f"Unsupported readout: {readout}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a frozen ENACT subspace on paired external prompts.")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--basis", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source", choices=("local", "hub"), required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--readout", choices=("yes_no", "rating_1_7"), required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()

    specs = load_model_specs()
    if arguments.model_id not in specs:
        raise KeyError(f"Unknown model: {arguments.model_id}")
    records = read_jsonl(arguments.data)
    if not records:
        raise ValueError("No external evaluation instances")
    basis_array = np.load(arguments.basis, allow_pickle=False)
    if basis_array.ndim != 2:
        raise ValueError("Subspace basis must have shape [hidden, rank]")
    basis = orthonormal_basis(torch.from_numpy(basis_array))
    model, tokenizer, metadata = load_model(
        specs[arguments.model_id], arguments.model_source, arguments.model_root,
        arguments.revision, arguments.device_map,
    )
    if basis.shape[0] != model.config.hidden_size:
        raise ValueError(
            f"Basis hidden size {basis.shape[0]} differs from model hidden size {model.config.hidden_size}"
        )

    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(records), arguments.batch_size):
        selected = records[start : start + arguments.batch_size]
        pairs = [prompt_pair(record) for record in selected]
        targets = [render_chat(tokenizer, specs[arguments.model_id], pair[0]) for pair in pairs]
        donors = [render_chat(tokenizer, specs[arguments.model_id], pair[1]) for pair in pairs]
        clean_logits, target_by_layer, _ = capture_states(model, tokenizer, targets, [arguments.layer])
        donor_logits, donor_by_layer, _ = capture_states(model, tokenizer, donors, [arguments.layer])
        target_states = target_by_layer[arguments.layer]
        donor_states = donor_by_layer[arguments.layer]
        projected = replace_projection(target_states, donor_states, basis)
        positions = torch.full((len(selected),), -1, dtype=torch.long)
        interface_logits = run_with_patches(
            model, tokenizer, targets, [PatchSpec(arguments.layer, positions, projected)],
        )
        full_logits = run_with_patches(
            model, tokenizer, targets, [PatchSpec(arguments.layer, positions, donor_states)],
        )
        scores = {
            "target": readout_scores(clean_logits, tokenizer, arguments.readout),
            "donor": readout_scores(donor_logits, tokenizer, arguments.readout),
            "interface": readout_scores(interface_logits, tokenizer, arguments.readout),
            "full_state": readout_scores(full_logits, tokenizer, arguments.readout),
        }
        for index, record in enumerate(selected):
            output_rows.append({
                "record_id": record["record_id"],
                "target": float(scores["target"][index]),
                "donor": float(scores["donor"][index]),
                "interface": float(scores["interface"][index]),
                "full_state": float(scores["full_state"][index]),
            })
    result = {
        "model": metadata,
        "layer": arguments.layer,
        "rank": basis.shape[1],
        "readout": arguments.readout,
        "basis_file": arguments.basis.name,
        "records": output_rows,
    }
    write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
