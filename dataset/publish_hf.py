from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import DatasetDict, load_from_disk
from huggingface_hub import HfApi


def main() -> int:
    parser = argparse.ArgumentParser(description="Publish validated MORALTRACE configs to Hugging Face.")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--dataset-card", type=Path, required=True)
    parser.add_argument("--croissant", type=Path, required=True)
    arguments = parser.parse_args()
    manifest = json.loads(arguments.manifest.read_text(encoding="utf-8"))
    for config_name, expected in manifest["configs"].items():
        dataset = load_from_disk(arguments.data_root / config_name)
        if not isinstance(dataset, DatasetDict):
            raise TypeError(f"{config_name} must contain named splits")
        observed = sum(len(split) for split in dataset.values())
        if observed != expected:
            raise ValueError(f"{config_name}: expected {expected}, found {observed}")
        dataset.push_to_hub(arguments.repo_id, config_name=config_name, revision=arguments.revision)
    api = HfApi()
    uploads = [
        (arguments.dataset_card, "README.md"),
        (arguments.croissant, "croissant.json"),
        (arguments.manifest, "moraltrace.json"),
    ]
    for local_path, repository_path in uploads:
        if not local_path.is_file():
            raise FileNotFoundError(local_path)
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=repository_path,
            repo_id=arguments.repo_id,
            repo_type="dataset",
            revision=arguments.revision,
            commit_message=f"Add {repository_path}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
