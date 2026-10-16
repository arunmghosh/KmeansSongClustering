"""
Rapid end-to-end integration smoke tests.
Verifies full experiment, trials averaging, custom parameters, and CLI smoke test runner.
"""

import unittest
from experiment import run_experiment, run_smoke_test
from kmeans import run_kmeans_trials
from preprocess import load_data, get_features
from runner import main as runner_main


class TestSmokeIntegration(unittest.TestCase):
    """Smoke test suite to ensure rapid end-to-end execution without regression."""

    @classmethod
    def setUpClass(cls):
        cls.data_path = "artist_data.xlsx"
        cls.df = load_data(cls.data_path)
        cls.X = get_features(cls.df)

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

    def test_smoke_trials_averaging(self):
        """Smoke test run_kmeans_trials multi-trial execution and averaging."""
        res = run_kmeans_trials(
            self.X,
            k=3,
            n_trials=3,
            max_iters=50,
            random_state=42,
        )
        self.assertEqual(res["k"], 3)
        self.assertEqual(res["centroids"].shape, (3, 3))
        self.assertEqual(len(res["assignments"]), len(self.X))
        self.assertGreater(res["score"], 0.0)
        self.assertEqual(len(res["trial_scores"]), 3)

    def test_smoke_custom_k_range(self):
        """Smoke test run_smoke_test helper function."""
        success = run_smoke_test(
            data_path=self.data_path,
            random_state=42,
        )
        self.assertTrue(success)

    def test_cli_smoke_flag(self):
        """Smoke test runner.py CLI entrypoint with --smoke-test flag."""
        exit_code = runner_main(["--smoke-test", "--data-path", self.data_path])
        self.assertEqual(exit_code, 0)


if __name__ == "__main__":
    unittest.main()
