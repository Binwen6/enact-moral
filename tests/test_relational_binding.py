from __future__ import annotations

import unittest

from experiments.relational_binding import cell_key, donor_indices


class RelationalBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.records = []
        graph = "allocation_scarce_resource_priority_034"
        setting = "normative_social"
        for affected in ("explicit_affected_role", "latent_affected_role"):
            for source in ("background_event_source", "intermediate_event_source"):
                for outcome in ("beneficial", "harmful"):
                    self.records.append({
                        "record_id": f"{graph}/{setting}/{affected}/{source}/{outcome}",
                        "graph_template_id": graph,
                        "semantic_setting": setting,
                        "manipulations": {
                            "affected_role_realization": affected,
                            "causal_connection": source,
                            "outcome_valence": outcome,
                        },
                    })

    def test_donors_change_declared_factors(self) -> None:
        donors = donor_indices(self.records)
        for index, record in enumerate(self.records):
            target = cell_key(record)
            relation = cell_key(self.records[int(donors["relation"][index])])
            valence = cell_key(self.records[int(donors["valence"][index])])
            joint = cell_key(self.records[int(donors["joint"][index])])
            self.assertEqual(target[:3], relation[:3])
            self.assertNotEqual(target[3], relation[3])
            self.assertEqual(target[4], relation[4])
            self.assertEqual(target[:4], valence[:4])
            self.assertNotEqual(target[4], valence[4])
            self.assertEqual(target[:3], joint[:3])
            self.assertNotEqual(target[3], joint[3])
            self.assertNotEqual(target[4], joint[4])


if __name__ == "__main__":
    unittest.main()
