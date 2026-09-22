from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_validation import load_manifest, validate_manifest  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Download a pinned MORALTRACE release from Hugging Face.")
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", action="append", dest="configs")
    arguments = parser.parse_args()

    manifest = load_manifest(REPO_ROOT / "data/manifests/moraltrace.json")
    validate_manifest(manifest)
    selected = arguments.configs or list(manifest["configs"])
    unknown = set(selected) - set(manifest["configs"])
    if unknown:
        raise ValueError(f"Unknown configs: {sorted(unknown)}")
    arguments.output.mkdir(parents=True, exist_ok=False)
    for config_name in selected:
        dataset = load_dataset(arguments.dataset_id, name=config_name, revision=arguments.revision)
        observed = sum(len(split) for split in dataset.values())
        expected = int(manifest["configs"][config_name])
        if observed != expected:
            raise ValueError(f"{config_name}: expected {expected}, found {observed}")
        config_root = arguments.output / config_name
        dataset.save_to_disk(config_root)
        jsonl_root = config_root / "jsonl"
        jsonl_root.mkdir()
        for split_name, split in dataset.items():
            split.to_json(jsonl_root / f"{split_name}.jsonl", orient="records", lines=True)
    (arguments.output / "download.json").write_text(
        json.dumps({"dataset_id": arguments.dataset_id, "revision": arguments.revision, "configs": selected}, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
