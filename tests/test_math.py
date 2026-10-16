"""
Mathematical correctness tests for Song Clustering algorithms.
Verifies Euclidean distance formulas, WCSS objective calculation,
Gini Impurity, and Hungarian centroid alignment & averaging.
"""

import unittest
import numpy as np
from kmeans import (
    euclidean_distance,
    objective_function,
    align_centroids,
    run_kmeans_trials,
)
from stats import compute_gini_impurity


class TestMathCalculations(unittest.TestCase):
    """Test suite for mathematical accuracy of formulas used in the project."""

    def test_euclidean_distance(self):
        """Test Euclidean distance calculation against known analytic values."""
        # 3D points: (0, 0, 0) and (1, 2, 2) -> sqrt(1 + 4 + 4) = sqrt(9) = 3.0
        p1 = np.array([0.0, 0.0, 0.0])
        p2 = np.array([1.0, 2.0, 2.0])

        dist = euclidean_distance(p1, p2, squared=False)
        self.assertAlmostEqual(dist, 3.0)

        sq_dist = euclidean_distance(p1, p2, squared=True)
        self.assertAlmostEqual(sq_dist, 9.0)

        # Pairwise distance matrix test:
        # A = [[0, 0, 0], [3, 4, 0]]
        # B = [[0, 0, 0], [1, 2, 2]]
        A = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
        B = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
        # Expected pairwise:
        # A[0] to B[0] = 0, A[0] to B[1] = 3
        # A[1] to B[0] = 5, A[1] to B[1] = sqrt((3-1)^2 + (4-2)^2 + (0-2)^2) = sqrt(4+4+4) = sqrt(12)
        pairwise = euclidean_distance(A, B, squared=False)
        self.assertAlmostEqual(pairwise[0, 0], 0.0)
        self.assertAlmostEqual(pairwise[0, 1], 3.0)
        self.assertAlmostEqual(pairwise[1, 0], 5.0)
        self.assertAlmostEqual(pairwise[1, 1], np.sqrt(12.0))

    def test_objective_function_wcss(self):
        """Test Within-Cluster Sum of Squares (WCSS) J = sum ||x - mu||^2 against manual calculation."""
        # Cluster 0 points: [1, 0, 0], [3, 0, 0] -> Centroid = [2, 0, 0]
        # Distances to centroid: (1-2)^2 = 1, (3-2)^2 = 1 -> Sum = 2.0
        # Cluster 1 points: [0, 4, 0], [0, 6, 0] -> Centroid = [0, 5, 0]
        # Distances to centroid: (4-5)^2 = 1, (6-5)^2 = 1 -> Sum = 2.0
        # Total expected WCSS = 2.0 + 2.0 = 4.0
        X = np.array([
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 6.0, 0.0],
        ])
        centroids = np.array([
            [2.0, 0.0, 0.0],
            [0.0, 5.0, 0.0],
        ])
        assignments = np.array([0, 0, 1, 1])

        computed_wcss = objective_function(X, centroids, assignments)
        self.assertAlmostEqual(computed_wcss, 4.0)

    def test_gini_impurity(self):
        """Test Gini Impurity G = 1 - sum(p_i^2) across pure, uniform, and custom distributions."""
        # 1. Pure cluster: all 10 items in cluster 0
        pure = [0] * 10
        self.assertAlmostEqual(compute_gini_impurity(pure), 0.0)

        # 2. Uniform across 4 clusters (k=4): p_i = 1/4 for all i
        # Gini = 1 - 4 * (1/4)^2 = 1 - 4/16 = 1 - 0.25 = 0.75
        uniform_4 = [0, 0, 1, 1, 2, 2, 3, 3]
        self.assertAlmostEqual(compute_gini_impurity(uniform_4), 0.75)

        # 3. Custom: 6 items in cluster 0, 2 items in cluster 1, 2 items in cluster 2 (total 10)
        # p_0 = 0.6 -> p_0^2 = 0.36
        # p_1 = 0.2 -> p_1^2 = 0.04
        # p_2 = 0.2 -> p_2^2 = 0.04
        # Sum = 0.44 -> Gini = 1 - 0.44 = 0.56
        custom = [0] * 6 + [1] * 2 + [2] * 2
        self.assertAlmostEqual(compute_gini_impurity(custom), 0.56)

    def test_centroid_averaging_and_alignment(self):
        """Test that Hungarian alignment aligns permuted centroid rows before averaging."""
        ref_centroids = np.array([
            [1.0, 1.0, 1.0],
            [5.0, 5.0, 5.0],
            [9.0, 9.0, 9.0],
        ])
        # Permuted order: row 2, row 0, row 1
        permuted_centroids = np.array([
            [9.1, 9.1, 9.1],
            [1.1, 1.1, 1.1],
            [5.1, 5.1, 5.1],
        ])

        aligned = align_centroids(ref_centroids, permuted_centroids)

        # Aligned rows should now match ref_centroids positions
        np.testing.assert_allclose(aligned[0], [1.1, 1.1, 1.1], atol=1e-5)
        np.testing.assert_allclose(aligned[1], [5.1, 5.1, 5.1], atol=1e-5)
        np.testing.assert_allclose(aligned[2], [9.1, 9.1, 9.1], atol=1e-5)

        # Average of ref and aligned should be ~ [1.05, 5.05, 9.05]
        avg = np.mean([ref_centroids, aligned], axis=0)
        expected_avg = np.array([
            [1.05, 1.05, 1.05],
            [5.05, 5.05, 5.05],
            [9.05, 9.05, 9.05],
        ])
        np.testing.assert_allclose(avg, expected_avg, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
