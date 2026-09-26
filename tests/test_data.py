from __future__ import annotations

import json
import unittest

from config import PROJECT_ROOT, load_model_specs, read_yaml
from data_validation import (
    load_manifest,
    validate_auxiliary_samples,
    validate_manifest,
    validate_release_metadata,
    validate_samples,
)
from io_utils import decode_instance, read_jsonl


class DataReleaseTests(unittest.TestCase):
    def test_released_row_is_decoded(self) -> None:
        instance = {"record_id": "graph/normative_social/cell", "record_type": "core_factorial"}
        row = {
            "record_id": instance["record_id"],
            "record_type": "core_factorial",
            "split": "train",
            "record_json": json.dumps(instance),
        }
        self.assertEqual(decode_instance(row), instance)
        self.assertEqual(decode_instance(instance), instance)

    def test_manifest_and_sample_coverage(self) -> None:
        manifest = load_manifest(PROJECT_ROOT / "data/manifests/moraltrace.json")
        validate_manifest(manifest)
        validate_samples(PROJECT_ROOT / "data/samples/samples.jsonl", set(manifest["configs"]))
        validate_release_metadata(
            PROJECT_ROOT / "dataset/croissant.json",
            PROJECT_ROOT / "dataset/DATASET_CARD.md",
            manifest,
        )
        validate_auxiliary_samples(
            PROJECT_ROOT / "data/samples/external_pairs.jsonl",
            PROJECT_ROOT / "data/samples/node_collapse.jsonl",
        )

    def test_participant_completion_quartet(self) -> None:
        records = read_jsonl(PROJECT_ROOT / "data/samples/participant_completion.jsonl")
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(set(record["units"]), {"A", "B", "C", "D"})
        self.assertTrue(record["meta"]["validated"])
        for unit in record["units"].values():
            self.assertIn(unit["scenario"], unit["full_prompt"])

    def test_manifest_is_valid_json(self) -> None:
        path = PROJECT_ROOT / "data/manifests/moraltrace.json"
        with path.open(encoding="utf-8") as handle:
            manifest = json.load(handle)
        self.assertEqual(manifest["total_records"], 51_177)

    def test_frozen_graph_sample(self) -> None:
        records = read_jsonl(PROJECT_ROOT / "data/samples/formal_graphs.jsonl")
        self.assertEqual(len(records), 1)
        graph = records[0]
        self.assertEqual(
            set(graph["settings"]),
            {"normative_social", "instrumental_social", "mechanistic_computational"},
        )
        self.assertTrue(graph["surface_realization_provenance"]["graph_frozen_before_surface_generation"])

    def test_paper_configuration_uses_declared_models(self) -> None:
        models = set(load_model_specs())
        experiments = read_yaml(PROJECT_ROOT / "configs/paper_experiments.yaml")
        self.assertLessEqual(set(experiments["patient_completion"]["layers"]), models)
        self.assertLessEqual(set(experiments["relational_binding"]["layers"]), models)
        self.assertLessEqual(set(experiments["compact_interface"]["layers"]), models)
        self.assertLessEqual(
            set(experiments["component_mediation"]["faithful_circuit_component_count"]),
            models,
        )
        self.assertLessEqual(
            set(experiments["component_mediation"]["feature_mediation_top4"]),
            models,
        )


if __name__ == "__main__":
    unittest.main()
