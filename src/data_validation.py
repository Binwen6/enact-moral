from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import load_dataset

from config import PROJECT_ROOT
from io_utils import read_jsonl


REQUIRED_SAMPLE_FIELDS = {"suite", "source_record_id", "split", "text"}


def load_manifest(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Invalid manifest: {path}")
    return value


def validate_manifest(manifest: dict[str, Any]) -> None:
    configs = manifest["configs"]
    declared = int(manifest["total_records"])
    counted = sum(int(count) for count in configs.values())
    if declared != 51_177 or counted != declared:
        raise ValueError(f"Expected 51,177 records, found manifest total {declared} and sum {counted}")
    if len(configs) != 9:
        raise ValueError(f"Expected nine dataset configs, found {len(configs)}")


def validate_samples(path: Path, expected_suites: set[str]) -> None:
    rows = read_jsonl(path)
    suites: set[str] = set()
    identifiers: set[tuple[str, str]] = set()
    for row in rows:
        missing = REQUIRED_SAMPLE_FIELDS - set(row)
        if missing:
            raise ValueError(f"Sample is missing fields {sorted(missing)}")
        if not str(row["text"]).strip():
            raise ValueError(f"Empty sample text: {row['source_record_id']}")
        key = (str(row["suite"]), str(row["source_record_id"]))
        if key in identifiers:
            raise ValueError(f"Duplicate sample: {key}")
        identifiers.add(key)
        suites.add(str(row["suite"]))
    if suites != expected_suites:
        raise ValueError(f"Sample suites differ from manifest: {sorted(suites ^ expected_suites)}")


def validate_hub(dataset_id: str, revision: str, manifest: dict[str, Any]) -> None:
    for config_name, expected in manifest["configs"].items():
        dataset = load_dataset(dataset_id, name=config_name, revision=revision)
        observed = sum(len(split) for split in dataset.values())
        if observed != expected:
            raise ValueError(f"{config_name}: expected {expected}, found {observed}")


def validate_participant_sample(path: Path) -> None:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one participant-completion sample, found {len(rows)}")
    record = rows[0]
    if set(record["units"]) != {"A", "B", "C", "D"}:
        raise ValueError(f"Incomplete participant quartet: {record.get('quartet_id')}")
    if not record.get("meta", {}).get("validated"):
        raise ValueError(f"Unvalidated participant quartet: {record.get('quartet_id')}")


def validate_graph_sample(path: Path) -> None:
    rows = read_jsonl(path)
    if len(rows) != 1:
        raise ValueError(f"Expected one formal graph sample, found {len(rows)}")
    record = rows[0]
    expected_settings = {"normative_social", "instrumental_social", "mechanistic_computational"}
    if set(record["settings"]) != expected_settings:
        raise ValueError("Formal graph sample must contain all three semantic settings")
    provenance = record.get("surface_realization_provenance", {})
    if not provenance.get("graph_frozen_before_surface_generation"):
        raise ValueError("Formal graph sample is missing its frozen-graph declaration")


def validate_release_metadata(croissant_path: Path, card_path: Path, manifest: dict[str, Any]) -> None:
    croissant = json.loads(croissant_path.read_text(encoding="utf-8"))
    if croissant.get("name") != "MORALTRACE" or croissant.get("version") != manifest["version"]:
        raise ValueError("Croissant name or version differs from the release manifest")
    if croissant.get("conformsTo") != "http://mlcommons.org/croissant/1.0":
        raise ValueError("Croissant metadata does not declare version 1.0")
    card = card_path.read_text(encoding="utf-8")
    missing = [config_name for config_name in manifest["configs"] if config_name not in card]
    if missing:
        raise ValueError(f"Dataset card is missing configurations: {missing}")
    if "CC BY 4.0" not in card:
        raise ValueError("Dataset card is missing the MORALTRACE license declaration")


def validate_auxiliary_samples(external_path: Path, node_path: Path) -> None:
    external_rows = read_jsonl(external_path)
    for row in external_rows:
        for field in ("record_id", "target_prompt", "donor_prompt"):
            if not isinstance(row.get(field), str) or not row[field].strip():
                raise ValueError(f"External pair is missing {field}")
    node_rows = read_jsonl(node_path)
    for row in node_rows:
        if set(row["cells"]) != {"direct", "mediated_controllable", "mediated_uncontrollable"}:
            raise ValueError(f"Incomplete node-collapse unit: {row.get('unit_id')}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate MORALTRACE manifests and representative samples.")
    parser.add_argument("--manifest", type=Path, default=PROJECT_ROOT / "data/manifests/moraltrace.json")
    parser.add_argument("--samples", type=Path, default=PROJECT_ROOT / "data/samples/samples.jsonl")
    parser.add_argument(
        "--participant-sample",
        type=Path,
        default=PROJECT_ROOT / "data/samples/participant_completion.jsonl",
    )
    parser.add_argument(
        "--graph-sample",
        type=Path,
        default=PROJECT_ROOT / "data/samples/formal_graphs.jsonl",
    )
    parser.add_argument("--croissant", type=Path, default=PROJECT_ROOT / "dataset/croissant.json")
    parser.add_argument("--dataset-card", type=Path, default=PROJECT_ROOT / "dataset/DATASET_CARD.md")
    parser.add_argument("--external-sample", type=Path, default=PROJECT_ROOT / "data/samples/external_pairs.jsonl")
    parser.add_argument("--node-sample", type=Path, default=PROJECT_ROOT / "data/samples/node_collapse.jsonl")
    parser.add_argument("--dataset-id")
    parser.add_argument("--revision")
    arguments = parser.parse_args()

    manifest = load_manifest(arguments.manifest)
    validate_manifest(manifest)
    validate_samples(arguments.samples, set(manifest["configs"]))
    validate_participant_sample(arguments.participant_sample)
    validate_graph_sample(arguments.graph_sample)
    validate_release_metadata(arguments.croissant, arguments.dataset_card, manifest)
    validate_auxiliary_samples(arguments.external_sample, arguments.node_sample)
    if (arguments.dataset_id is None) != (arguments.revision is None):
        parser.error("--dataset-id and --revision must be supplied together")
    if arguments.dataset_id:
        validate_hub(arguments.dataset_id, arguments.revision, manifest)
    print(json.dumps({"status": "ok", "records": manifest["total_records"], "configs": 9}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
