"""
Integration and pipeline tests for experiment execution and stage outputs.
Verifies preprocess loading, experiment result structure,
stage metrics, and versatility & album trajectory shifts.
"""

import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from preprocess import (
    load_data,
    get_features,
    FEATURE_COLUMNS,
)
from stats import (
    artist_cluster_distribution,
    artist_feature_averages,
    album_shifts,
    compute_versatility_scores,
    compare_centroids_to_artist_averages,
)
from experiment import run_experiment


class TestExperimentPipeline(unittest.TestCase):
    """Test suite for the experiment pipeline and output structures."""

    @classmethod
    def setUpClass(cls):
        cls.data_path = "artist_data.xlsx"
        cls.df = load_data(cls.data_path)

    def test_preprocess_loading(self):
        """Test loading artist_data.xlsx, verifying feature scales, and schema."""
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

        # Verify all 7 artists are present
        self.assertEqual(len(self.df["Artist"].unique()), 7)

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
        """Test stage 3 metrics computation on known assignments."""
        assignments = np.zeros(len(self.df), dtype=int)
        assignments[:50] = 1

        crosstab = artist_cluster_distribution(self.df, assignments)
        self.assertIn("Total", crosstab.columns)
        self.assertIn("Total", crosstab.index)

        means = artist_feature_averages(self.df)
        self.assertEqual(len(means), 7)
        self.assertEqual(list(means.columns), FEATURE_COLUMNS)

        # Comparison of dummy centroids to artist averages
        centroids = np.array([[2.0, 3.0, 4.0], [5.0, 5.0, 5.0]])
        comp = compare_centroids_to_artist_averages(centroids, means)
        self.assertIn("distance_matrix", comp)
        self.assertEqual(len(comp["closest_artist_per_cluster"]), 2)
        self.assertEqual(len(comp["closest_cluster_per_artist"]), 7)

    def test_versatility_and_album_shifts(self):
        """Test stage 4 versatility scores and album center shifts."""
        assignments = np.zeros(len(self.df), dtype=int)
        assignments[:50] = 1

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
        # Shift distances must be non-negative
        self.assertTrue((shifts["Shift_Distance"] >= 0).all())


if __name__ == "__main__":
    unittest.main()
