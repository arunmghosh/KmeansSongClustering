"""
Integration and pipeline tests for experiment execution and stage outputs.
Verifies preprocess loading, filtering, experiment result structure,
stage metrics, and phase evaluations.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from preprocess import (
    load_data,
    get_features,
    filter_phase1,
    filter_phase2,
    FEATURE_COLUMNS,
    PHASE1_ARTISTS,
    PHASE2_ALBUMS,
)
from stats import (
    artist_cluster_distribution,
    artist_feature_averages,
    album_shifts,
    compute_versatility_scores,
    evaluate_cluster_classification,
)
from experiment import run_experiment, run_phase1, run_phase2


class TestExperimentPipeline(unittest.TestCase):
    """Test suite for the experiment pipeline and output structures."""

    @classmethod
    def setUpClass(cls):
        cls.data_path = "artist_data.xlsx"
        cls.df = load_data(cls.data_path)

    def test_preprocess_loading_and_filtering(self):
        """Test loading artist_data.xlsx, verifying feature scales, and phase filtering."""
        self.assertEqual(len(self.df), 167)
        self.assertEqual(
            list(self.df.columns),
            ["Song", "Artist", "Album", "Energy", "Positivity", "Introspectiveness"],
        )

        # Verify features are kept on original 0-7 scale without scaling
        for col in FEATURE_COLUMNS:
            self.assertGreaterEqual(self.df[col].min(), 0)
            self.assertLessEqual(self.df[col].max(), 7)

        X = get_features(self.df)
        self.assertEqual(X.shape, (167, 3))
        self.assertEqual(X.dtype, np.float64)

        # Phase 1: Olivia Rodrigo (24) + Gracie Abrams (32) = 56 songs
        p1_df = filter_phase1(self.df)
        self.assertEqual(len(p1_df), 56)
        self.assertTrue(set(p1_df["Artist"].unique()).issubset(set(PHASE1_ARTISTS)))

        # Phase 2: 6 albums
        p2_df = filter_phase2(self.df)
        self.assertEqual(len(p2_df["Album"].unique()), 6)
        self.assertTrue(set(p2_df["Album"].unique()).issubset(set(PHASE2_ALBUMS)))

    def test_experiment_output_structure(self):
        """Test that run_experiment produces a dictionary with all required stages and keys."""
        res = run_experiment(
            data_path=self.data_path,
            min_k=2,
            max_k=3,
            n_trials=2,
            random_state=42,
        )

        expected_keys = [
            "dataset_shape",
            "k_range",
            "k_scores",
            "optimal_k",
            "optimal_score",
            "k_models",
            "stage2_inter_cluster_distances",
            "stage3_artist_crosstab",
            "stage3_artist_means",
            "stage3_centroid_comparison",
            "stage4_versatility",
            "stage4_album_shifts",
        ]
        for key in expected_keys:
            self.assertIn(key, res)

        self.assertIn(res["optimal_k"], [2, 3])
        self.assertIsInstance(res["stage4_versatility"], pd.DataFrame)
        self.assertIsInstance(res["stage4_album_shifts"], pd.DataFrame)

    def test_stage_metrics_computation(self):
        """Test stage 3 and stage 4 metrics computation on known assignments."""
        assignments = np.zeros(len(self.df), dtype=int)
        assignments[:50] = 1

        crosstab = artist_cluster_distribution(self.df, assignments)
        self.assertIn("Total", crosstab.columns)
        self.assertIn("Total", crosstab.index)

        means = artist_feature_averages(self.df)
        self.assertEqual(len(means), 7)
        self.assertEqual(list(means.columns), FEATURE_COLUMNS)

        versatility = compute_versatility_scores(self.df, assignments, k=2)
        self.assertEqual(len(versatility), 7)
        self.assertTrue(
            {"Artist", "Song_Count", "Gini_Impurity", "Normalized_Gini"}.issubset(
                set(versatility.columns)
            )
        )

        shifts = album_shifts(self.df)
        self.assertEqual(len(shifts), 7)
        self.assertTrue(
            {"Artist", "Album_1", "Album_2", "Shift_Distance"}.issubset(
                set(shifts.columns)
            )
        )

    def test_phase_evaluations(self):
        """Test that Phase 1 and Phase 2 pipelines compute valid classification metrics."""
        # Phase 1 test
        p1 = run_phase1(
            data_path=self.data_path,
            min_k=2,
            max_k=3,
            n_trials=2,
            random_state=42,
        )
        self.assertEqual(p1["phase"], 1)
        self.assertEqual(p1["num_samples"], 56)
        self.assertGreaterEqual(p1["best_accuracy"], 0.0)
        self.assertLessEqual(p1["best_accuracy"], 1.0)

        # Phase 2 test
        p2 = run_phase2(
            data_path=self.data_path,
            min_k=3,
            max_k=4,
            n_trials=2,
            random_state=42,
        )
        self.assertEqual(p2["phase"], 2)
        self.assertGreater(p2["num_samples"], 0)
        self.assertGreaterEqual(p2["best_accuracy"], 0.0)
        self.assertLessEqual(p2["best_accuracy"], 1.0)


if __name__ == "__main__":
    unittest.main()
