from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as functional

from config import load_model_specs
from interventions import HiddenStatePatch, LastTokenWrite, PatchSpec, capture_states
from io_utils import read_jsonl, write_json
from modeling import encode, input_device, load_model, render_chat
from readouts import single_token_id
from relational_binding import donor_indices, intention_prompt, rendered_segment_positions, select_complete_graphs

IDENTITY_TOLERANCE = 1e-3


@dataclass
class Graphs:
    records: list[dict[str, Any]]
    prompts: list[str]
    donors: torch.Tensor
    lengths: torch.Tensor
    # Distance from the sequence end, so positions survive any padding width
    relation_from_end: torch.Tensor
    valence_from_end: torch.Tensor
    relation_states: torch.Tensor
    valence_states: torch.Tensor


def batches(count: int, size: int) -> list[torch.Tensor]:
    return [torch.arange(start, min(start + size, count)) for start in range(0, count, size)]


def prepare(
    records: list[dict[str, Any]],
    model: Any,
    tokenizer: Any,
    spec: Any,
    arguments: argparse.Namespace,
) -> Graphs:
    raw_prompts = [intention_prompt(record) for record in records]
    prompts = [render_chat(tokenizer, spec, prompt) for prompt in raw_prompts]
    lengths = torch.tensor([len(tokenizer(prompt)["input_ids"]) for prompt in prompts])
    width = int(lengths.max())
    relation_from_end = width - rendered_segment_positions(tokenizer, records, raw_prompts, prompts, "causal_statement")
    valence_from_end = width - rendered_segment_positions(tokenizer, records, raw_prompts, prompts, "outcome")
    relation_states: list[torch.Tensor] = []
    valence_states: list[torch.Tensor] = []
    for rows in batches(len(records), arguments.batch_size):
        chunk = [prompts[int(row)] for row in rows]
        chunk_width = int(lengths[rows].max())
        _, relation, _ = capture_states(
            model, tokenizer, chunk, [arguments.relation_layer], chunk_width - relation_from_end[rows],
        )
        _, valence, _ = capture_states(
            model, tokenizer, chunk, [arguments.valence_layer], chunk_width - valence_from_end[rows],
        )
        relation_states.append(relation[arguments.relation_layer])
        valence_states.append(valence[arguments.valence_layer])
    return Graphs(
        records=records,
        prompts=prompts,
        donors=donor_indices(records)["joint"],
        lengths=lengths,
        relation_from_end=relation_from_end,
        valence_from_end=valence_from_end,
        relation_states=torch.cat(relation_states),
        valence_states=torch.cat(valence_states),
    )


def run_rows(
    model: Any,
    tokenizer: Any,
    data: Graphs,
    rows: torch.Tensor,
    tap: LastTokenWrite,
    arguments: argparse.Namespace,
    tokens: tuple[int, int],
    joint: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Final-token residual at the interface layer and the readout margin of one batch
    batch = encode(tokenizer, [data.prompts[int(row)] for row in rows], input_device(model))
    width = batch["input_ids"].shape[1]
    patches: list[PatchSpec] = []
    if joint:
        donor_rows = data.donors[rows]
        patches = [
            PatchSpec(arguments.relation_layer, width - data.relation_from_end[rows], data.relation_states[donor_rows]),
            PatchSpec(arguments.valence_layer, width - data.valence_from_end[rows], data.valence_states[donor_rows]),
        ]
    tap.mode = "capture"
    with HiddenStatePatch(model, patches), torch.no_grad():
        logits = model(**batch, use_cache=False).logits[:, -1, :]
    tap.mode = "off"
    return tap.captured, (logits[:, tokens[0]] - logits[:, tokens[1]]).float()


def write_rows(
    model: Any,
    tokenizer: Any,
    data: Graphs,
    rows: torch.Tensor,
    tap: LastTokenWrite,
    source: torch.Tensor,
    basis: torch.Tensor | None,
    tokens: tuple[int, int],
    train: bool,
) -> torch.Tensor:
    # Clean run whose final-token residual moves toward the source inside the basis
    batch = encode(tokenizer, [data.prompts[int(row)] for row in rows], input_device(model))
    tap.mode = "write"
    tap.source = source
    tap.basis = basis
    try:
        with torch.set_grad_enabled(train):
            logits = model(**batch, use_cache=False).logits[:, -1, :]
    finally:
        tap.mode = "off"
        tap.source = None
        tap.basis = None
    return (logits[:, tokens[0]] - logits[:, tokens[1]]).float()


def train_basis(
    model: Any,
    tokenizer: Any,
    data: Graphs,
    tap: LastTokenWrite,
    arguments: argparse.Namespace,
    tokens: tuple[int, int],
    seed: int,
    shuffled: bool,
) -> tuple[torch.Tensor, list[float]]:
    # The basis regresses the clean run onto the margin of the joint-donor run.
    # With shuffled supervision each source residual is paired with a wrong example.
    hidden = int(model.config.hidden_size)
    generator = torch.Generator().manual_seed(seed)
    raw = (torch.randn(hidden, arguments.rank, generator=generator) / hidden ** 0.5)
    raw = raw.to(input_device(model)).requires_grad_(True)
    optimizer = torch.optim.Adam([raw], lr=arguments.learning_rate)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(data.records))
    cursor = 0
    losses: list[float] = []
    for _ in range(arguments.train_steps):
        if cursor + arguments.batch_size > len(order):
            order = rng.permutation(len(data.records))
            cursor = 0
        rows = torch.from_numpy(order[cursor : cursor + arguments.batch_size].copy())
        cursor += arguments.batch_size
        source, target = run_rows(model, tokenizer, data, rows, tap, arguments, tokens, joint=True)
        if shuffled:
            source = source[torch.from_numpy(rng.permutation(len(rows))).to(source.device)]
        margin = write_rows(model, tokenizer, data, rows, tap, source, torch.linalg.qr(raw).Q, tokens, train=True)
        loss = functional.mse_loss(margin, target.to(margin.device))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    with torch.no_grad():
        return torch.linalg.qr(raw).Q.detach().cpu(), losses


def evaluate(
    model: Any,
    tokenizer: Any,
    data: Graphs,
    tap: LastTokenWrite,
    arguments: argparse.Namespace,
    tokens: tuple[int, int],
    bases: dict[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], float]:
    names = ["clean", "joint", "full_state", *bases]
    scores = {name: torch.zeros(len(data.records)) for name in names}
    identity_error = 0.0
    for rows in batches(len(data.records), arguments.batch_size):
        clean_state, clean_margin = run_rows(model, tokenizer, data, rows, tap, arguments, tokens, joint=False)
        joint_state, joint_margin = run_rows(model, tokenizer, data, rows, tap, arguments, tokens, joint=True)
        scores["clean"][rows] = clean_margin.cpu()
        scores["joint"][rows] = joint_margin.cpu()
        # Writing the clean state back into the clean run must leave the readout unchanged
        identity = write_rows(model, tokenizer, data, rows, tap, clean_state, None, tokens, train=False)
        identity_error = max(identity_error, float((identity - clean_margin).abs().max()))
        scores["full_state"][rows] = write_rows(
            model, tokenizer, data, rows, tap, joint_state, None, tokens, train=False,
        ).cpu()
        for name, basis in bases.items():
            scores[name][rows] = write_rows(
                model, tokenizer, data, rows, tap, joint_state, basis.to(joint_state.device), tokens, train=False,
            ).cpu()
    if identity_error > IDENTITY_TOLERANCE:
        raise RuntimeError(f"Writing the clean state changed the readout by {identity_error:.6f}")
    return scores, identity_error


def progress(clean: torch.Tensor, scores: torch.Tensor, desired: torch.Tensor) -> torch.Tensor:
    # Reduction of the distance to the counterfactual donor's clean margin
    return (clean - desired).abs() - (scores - desired).abs()


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a low-rank ENACT subspace on train graphs and evaluate it on validation graphs.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--evaluation-data", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--model-source", choices=("local", "hub"), required=True)
    parser.add_argument("--model-root", type=Path)
    parser.add_argument("--revision")
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--setting", default="normative_social")
    parser.add_argument("--layer", type=int, required=True)
    parser.add_argument("--relation-layer", type=int, required=True)
    parser.add_argument("--valence-layer", type=int, required=True)
    parser.add_argument("--rank", type=int, default=1)
    parser.add_argument("--train-graphs", type=int, default=1)
    parser.add_argument("--evaluation-graphs", type=int, default=1)
    parser.add_argument("--train-steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=1e-2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument("--shuffled-control", action="store_true")
    parser.add_argument("--basis-output", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.rank < 1:
        raise ValueError("Rank must be positive")
    if arguments.layer <= max(arguments.relation_layer, arguments.valence_layer):
        raise ValueError("The interface layer must lie above both source layers")

    specs = load_model_specs()
    if arguments.model_id not in specs:
        raise KeyError(f"Unknown model: {arguments.model_id}")
    spec = specs[arguments.model_id]
    train = select_complete_graphs(read_jsonl(arguments.train_data), "train", arguments.setting, arguments.train_graphs)
    evaluation = select_complete_graphs(
        read_jsonl(arguments.evaluation_data), "validation", arguments.setting, arguments.evaluation_graphs,
    )
    model, tokenizer, metadata = load_model(
        spec, arguments.model_source, arguments.model_root, arguments.revision, arguments.device_map,
    )
    # Only the basis is trained, so the weights carry no gradient
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    tokens = (single_token_id(tokenizer, "Yes"), single_token_id(tokenizer, "No"))
    train_data = prepare(train, model, tokenizer, spec, arguments)
    evaluation_data = prepare(evaluation, model, tokenizer, spec, arguments)

    seed = arguments.seed + 8000 + 100 * arguments.layer + arguments.rank
    with LastTokenWrite(model, arguments.layer) as tap:
        basis, losses = train_basis(model, tokenizer, train_data, tap, arguments, tokens, seed, shuffled=False)
        bases = {"interface": basis}
        shuffled_losses: list[float] | None = None
        if arguments.shuffled_control:
            bases["shuffled"], shuffled_losses = train_basis(
                model, tokenizer, train_data, tap, arguments, tokens, seed + 50, shuffled=True,
            )
        scores, identity_error = evaluate(model, tokenizer, evaluation_data, tap, arguments, tokens, bases)
    if arguments.basis_output is not None:
        arguments.basis_output.parent.mkdir(parents=True, exist_ok=True)
        np.save(arguments.basis_output, basis.numpy())

    desired = scores["clean"].index_select(0, evaluation_data.donors)
    gains = {name: progress(scores["clean"], scores[name], desired) for name in scores if name != "clean"}
    joint_gain = float(gains["joint"].mean())
    result = {
        "model": metadata,
        "setting": arguments.setting,
        "layer": arguments.layer,
        "relation_layer": arguments.relation_layer,
        "valence_layer": arguments.valence_layer,
        "rank": arguments.rank,
        "train_steps": arguments.train_steps,
        "learning_rate": arguments.learning_rate,
        "batch_size": arguments.batch_size,
        "seed": seed,
        "basis_file": arguments.basis_output.name if arguments.basis_output is not None else None,
        "train_graphs": arguments.train_graphs,
        "evaluation_graphs": arguments.evaluation_graphs,
        "training_loss": {"first": losses[0], "last": losses[-1]},
        "identity_max_logit_difference": identity_error,
        "mean_joint_progress": joint_gain,
        "mean_full_state_progress": float(gains["full_state"].mean()),
        "mean_interface_progress": float(gains["interface"].mean()),
        # The ratio is undefined without a positive joint effect
        "recovery_ratio": float(gains["interface"].mean()) / joint_gain if joint_gain > 0 else None,
        "records": [
            {
                "record_id": record["record_id"],
                "desired": float(desired[index]),
                **{name: float(values[index]) for name, values in scores.items()},
            }
            for index, record in enumerate(evaluation)
        ],
    }
    if shuffled_losses is not None:
        result["mean_shuffled_progress"] = float(gains["shuffled"].mean())
        result["shuffled_training_loss"] = {"first": shuffled_losses[0], "last": shuffled_losses[-1]}
    write_json(arguments.output, result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
