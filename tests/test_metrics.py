from __future__ import annotations

import unittest

from config import PROJECT_ROOT
from paper_analysis import validate_metrics


class PaperMetricTests(unittest.TestCase):
    def test_released_metric_contracts(self) -> None:
        counts = validate_metrics(PROJECT_ROOT / "results/paper_metrics")
        self.assertEqual(counts["table2_models"], 8)
        self.assertEqual(counts["table3_models"], 4)
        self.assertEqual(counts["scale_points"], 7)


if __name__ == "__main__":
    unittest.main()
