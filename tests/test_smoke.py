"""
Rapid end-to-end integration smoke tests.
Verifies full experiment, Phase 1, Phase 2, and CLI smoke test runner.
"""

import unittest
from experiment import run_experiment, run_phase1, run_phase2, run_smoke_test
from runner import main as runner_main


class TestSmokeIntegration(unittest.TestCase):
    """Smoke test suite to ensure rapid end-to-end execution without regression."""

    @classmethod
    def setUpClass(cls):
        cls.data_path = "artist_data.xlsx"

    def test_smoke_full_experiment(self):
        """Smoke test full experiment pipeline on small K range and minimal trials."""
        res = run_experiment(
            data_path=self.data_path,
            min_k=2,
            max_k=3,
            n_trials=2,
            random_state=42,
        )
        self.assertIsNotNone(res)
        self.assertIn("optimal_k", res)
        self.assertIn(res["optimal_k"], [2, 3])
        self.assertGreater(len(res["k_scores"]), 0)

    def test_smoke_phase1(self):
        """Smoke test Phase 1 binary classification pipeline."""
        p1 = run_phase1(
            data_path=self.data_path,
            min_k=2,
            max_k=3,
            n_trials=2,
            random_state=42,
        )
        self.assertEqual(p1["phase"], 1)
        self.assertIn("best_accuracy", p1)
        self.assertGreaterEqual(p1["best_accuracy"], 0.0)

    def test_smoke_phase2(self):
        """Smoke test Phase 2 multinomial album classification pipeline."""
        p2 = run_phase2(
            data_path=self.data_path,
            min_k=3,
            max_k=4,
            n_trials=2,
            random_state=42,
        )
        self.assertEqual(p2["phase"], 2)
        self.assertIn("best_accuracy", p2)
        self.assertGreaterEqual(p2["best_accuracy"], 0.0)

    def test_cli_smoke_flag(self):
        """Smoke test runner.py CLI entrypoint with --smoke-test flag."""
        exit_code = runner_main(["--smoke-test", "--data-path", self.data_path])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
