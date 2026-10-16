"""
Unit tests for K-means model logic.
Verifies shapes, convergence, edge cases, and deterministic seeding.
"""

import unittest
import numpy as np
from kmeans import KMeans, assign_clusters, update_centroids, initialize_centroids


class TestKMeansModels(unittest.TestCase):
    """Test suite for KMeans model implementation."""

    def test_fit_and_predict_shapes(self):
        """Test that KMeans produces correct centroid and assignment shapes."""
        rng = np.random.RandomState(42)
        X = rng.rand(50, 3) * 7.0  # (50, 3) feature matrix
        k = 4

        model = KMeans(k=k, random_state=42)
        model.fit(X)

        self.assertIsNotNone(model.centroids)
        self.assertEqual(model.centroids.shape, (k, 3))
        self.assertEqual(model.labels_.shape, (50,))
        self.assertTrue(np.all(model.labels_ >= 0))
        self.assertTrue(np.all(model.labels_ < k))

        # Prediction on new samples
        X_new = rng.rand(10, 3) * 7.0
        preds = model.predict(X_new)
        self.assertEqual(preds.shape, (10,))
        self.assertTrue(np.all(preds >= 0))
        self.assertTrue(np.all(preds < k))

    def test_convergence(self):
        """Test that KMeans converges within max_iters and produces finite inertia."""
        rng = np.random.RandomState(123)
        # Create 3 clearly separated clusters
        c1 = rng.randn(20, 3) + np.array([1.0, 1.0, 1.0])
        c2 = rng.randn(20, 3) + np.array([5.0, 5.0, 5.0])
        c3 = rng.randn(20, 3) + np.array([9.0, 1.0, 5.0])
        X = np.vstack([c1, c2, c3])

        model = KMeans(k=3, max_iters=100, random_state=42)
        model.fit(X)

        self.assertLessEqual(model.n_iter_, 100)
        self.assertIsNotNone(model.inertia_)
        self.assertTrue(np.isfinite(model.inertia_))
        self.assertGreater(model.inertia_, 0.0)

    def test_empty_cluster_handling(self):
        """Test that update_centroids gracefully handles empty clusters without NaNs."""
        X = np.array([[1.0, 2.0, 3.0], [1.1, 2.1, 3.1], [1.2, 2.2, 3.2]])
        # Artificially assign all points to cluster 0 in a k=3 setting
        assignments = np.array([0, 0, 0])
        current_centroids = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 0.0], [5.0, 5.0, 5.0]])

        new_centroids = update_centroids(
            X, assignments, k=3, current_centroids=current_centroids
        )

        self.assertEqual(new_centroids.shape, (3, 3))
        self.assertFalse(np.isnan(new_centroids).any())
        # Cluster 0 should be the mean of X
        np.testing.assert_allclose(new_centroids[0], np.mean(X, axis=0))

    def test_deterministic_seed(self):
        """Test that setting random_state reproduces exact centroids and assignments."""
        rng = np.random.RandomState(99)
        X = rng.rand(40, 3) * 7.0

        model1 = KMeans(k=5, random_state=42).fit(X)
        model2 = KMeans(k=5, random_state=42).fit(X)

        np.testing.assert_allclose(model1.centroids, model2.centroids)
        np.testing.assert_array_equal(model1.labels_, model2.labels_)
        self.assertAlmostEqual(model1.inertia_, model2.inertia_)


if __name__ == "__main__":
    unittest.main()
