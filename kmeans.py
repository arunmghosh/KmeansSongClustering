"""
K-means clustering algorithm implementation.
Supports custom distance calculations, WCSS objective scoring,
centroid initialization, convergence checks, Hungarian centroid alignment,
and multi-trial centroid averaging as specified in the experiment design.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
from scipy.optimize import linear_sum_assignment


def euclidean_distance(
    a: np.ndarray, b: np.ndarray, squared: bool = False
) -> Union[float, np.ndarray]:
    """Compute the Euclidean distance (or squared Euclidean distance) between points.

    Supports:
        - Two 1D vectors: returns scalar distance
        - 2D matrix (N, D) and 1D vector (D,): returns (N,) distances
        - Two 2D matrices (N, D) and (M, D): returns (N, M) pairwise distances
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    if a.ndim == 1 and b.ndim == 1:
        sq_dist = np.sum((a - b) ** 2)
        return float(sq_dist if squared else np.sqrt(sq_dist))

    if a.ndim == 2 and b.ndim == 1:
        sq_dist = np.sum((a - b) ** 2, axis=1)
        return sq_dist if squared else np.sqrt(sq_dist)

    if a.ndim == 1 and b.ndim == 2:
        sq_dist = np.sum((a - b) ** 2, axis=1)
        return sq_dist if squared else np.sqrt(sq_dist)

    if a.ndim == 2 and b.ndim == 2:
        # Pairwise distance matrix of shape (N, M)
        # Using ||a - b||^2 = ||a||^2 + ||b||^2 - 2 * a . b
        diff = a[:, np.newaxis, :] - b[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=-1)
        # Numerical stability clip
        sq_dist = np.maximum(sq_dist, 0.0)
        return sq_dist if squared else np.sqrt(sq_dist)

    raise ValueError(f"Unsupported dimensions: a.ndim={a.ndim}, b.ndim={b.ndim}")


def objective_function(
    X: np.ndarray, centroids: np.ndarray, assignments: np.ndarray
) -> float:
    """Compute the K-means objective function (Within-Cluster Sum of Squares, WCSS):

    J = sum_{i=1}^k sum_{x in C_i} ||x - mu_i||^2

    Args:
        X: Data matrix of shape (N, D)
        centroids: Centroid matrix of shape (K, D)
        assignments: Cluster index array of shape (N,)

    Returns:
        float: Total within-cluster squared Euclidean distance.
    """
    X = np.asarray(X, dtype=np.float64)
    centroids = np.asarray(centroids, dtype=np.float64)
    assignments = np.asarray(assignments, dtype=np.int64)

    total_wcss = 0.0
    for k_idx in range(len(centroids)):
        cluster_mask = assignments == k_idx
        if np.any(cluster_mask):
            pts = X[cluster_mask]
            sq_diff = np.sum((pts - centroids[k_idx]) ** 2)
            total_wcss += float(sq_diff)
    return total_wcss


def assign_clusters(X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """Assign each data point to its nearest centroid using Euclidean distance.

    Args:
        X: Data matrix of shape (N, D)
        centroids: Centroid matrix of shape (K, D)

    Returns:
        np.ndarray: 1D array of cluster indices (length N)
    """
    # Compute squared distances to all centroids: shape (N, K)
    distances = euclidean_distance(X, centroids, squared=True)
    return np.argmin(distances, axis=1).astype(np.int64)


def update_centroids(
    X: np.ndarray,
    assignments: np.ndarray,
    k: int,
    current_centroids: Optional[np.ndarray] = None,
    rng: Optional[np.random.RandomState] = None,
) -> np.ndarray:
    """Recalculate centroids as the arithmetic mean of points in each cluster.

    If a cluster is empty, reinitializes it with the point furthest from any centroid.

    Args:
        X: Data matrix of shape (N, D)
        assignments: Cluster index array of shape (N,)
        k: Number of clusters
        current_centroids: Previous centroids (for fallback/furthest point calculation)
        rng: Random number generator for tie-breaking

    Returns:
        np.ndarray: Updated centroid matrix of shape (k, D)
    """
    N, D = X.shape
    new_centroids = np.zeros((k, D), dtype=np.float64)

    for i in range(k):
        cluster_mask = assignments == i
        if np.any(cluster_mask):
            new_centroids[i] = np.mean(X[cluster_mask], axis=0)
        else:
            # Handle empty cluster: pick the point with the highest distance to existing centroids
            if current_centroids is not None and len(current_centroids) == k:
                dists = np.min(euclidean_distance(X, current_centroids, squared=True), axis=1)
                furthest_idx = int(np.argmax(dists))
                new_centroids[i] = X[furthest_idx]
            else:
                rand_idx = (
                    rng.randint(0, N) if rng is not None else np.random.randint(0, N)
                )
                new_centroids[i] = X[rand_idx]

    return new_centroids


def initialize_centroids(
    X: np.ndarray, k: int, rng: np.random.RandomState
) -> np.ndarray:
    """Randomly select k distinct data points as initial centroids."""
    N = len(X)
    if k > N:
        raise ValueError(f"Cannot initialize {k} clusters with only {N} points.")
    indices = rng.choice(N, size=k, replace=False)
    return X[indices].copy().astype(np.float64)


def align_centroids(
    ref_centroids: np.ndarray, target_centroids: np.ndarray
) -> np.ndarray:
    """Align target_centroids to match ref_centroids ordering using the Hungarian

    algorithm (linear sum assignment) on Euclidean distance.

    Args:
        ref_centroids: Reference centroids (K, D)
        target_centroids: Centroids to reorder (K, D)

    Returns:
        np.ndarray: Reordered target centroids of shape (K, D)
    """
    cost_matrix = euclidean_distance(ref_centroids, target_centroids, squared=False)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    # col_ind maps ref index i to target index col_ind[i]
    aligned = np.zeros_like(target_centroids)
    aligned[row_ind] = target_centroids[col_ind]
    return aligned


def compute_inter_cluster_distances(centroids: np.ndarray) -> Dict[str, Any]:
    """Compute pairwise Euclidean distances between cluster centroids.

    Returns:
        dict containing:
            - 'distance_matrix': (K, K) pairwise distance matrix
            - 'min_distance': minimum distance between two distinct centroids
            - 'mean_distance': average distance between distinct centroids
            - 'max_distance': maximum distance between two centroids
    """
    K = len(centroids)
    if K <= 1:
        return {
            "distance_matrix": np.zeros((K, K)),
            "min_distance": 0.0,
            "mean_distance": 0.0,
            "max_distance": 0.0,
        }

    dist_mat = euclidean_distance(centroids, centroids, squared=False)
    # Extract off-diagonal elements
    triu_indices = np.triu_indices(K, k=1)
    pairwise = dist_mat[triu_indices]

    return {
        "distance_matrix": dist_mat,
        "min_distance": float(np.min(pairwise)),
        "mean_distance": float(np.mean(pairwise)),
        "max_distance": float(np.max(pairwise)),
    }


class KMeans:
    """K-means clustering model."""

    def __init__(
        self,
        k: int = 8,
        max_iters: int = 300,
        tol: float = 1e-4,
        random_state: Optional[int] = None,
    ):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids: Optional[np.ndarray] = None
        self.labels_: Optional[np.ndarray] = None
        self.inertia_: Optional[float] = None  # WCSS objective score
        self.n_iter_: int = 0

    def fit(self, X: np.ndarray) -> "KMeans":
        """Fit K-means model to data matrix X."""
        X = np.asarray(X, dtype=np.float64)
        rng = np.random.RandomState(self.random_state)
        centroids = initialize_centroids(X, self.k, rng)

        for iteration in range(self.max_iters):
            assignments = assign_clusters(X, centroids)
            new_centroids = update_centroids(
                X, assignments, self.k, current_centroids=centroids, rng=rng
            )

            shift = np.max(np.linalg.norm(new_centroids - centroids, axis=1))
            centroids = new_centroids
            self.n_iter_ = iteration + 1

            if shift < self.tol:
                break

        self.centroids = centroids
        self.labels_ = assign_clusters(X, centroids)
        self.inertia_ = objective_function(X, self.centroids, self.labels_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict cluster index for each sample in X."""
        if self.centroids is None:
            raise RuntimeError("Model has not been fitted yet.")
        return assign_clusters(np.asarray(X, dtype=np.float64), self.centroids)

    def score(self, X: np.ndarray) -> float:
        """Compute the K-means objective function score (WCSS) on X."""
        X = np.asarray(X, dtype=np.float64)
        assignments = self.predict(X)
        return objective_function(X, self.centroids, assignments)


def run_kmeans_trials(
    X: np.ndarray,
    k: int,
    n_trials: int = 10,
    max_iters: int = 300,
    tol: float = 1e-4,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute multiple random initialization trials for a given K.

    As described in the experiment design:
    Each value of K receives n_trials of random centroid initializations.
    The final positions of the centroids are aligned and averaged across the trials
    to arrive at the final centroids for each K-value.
    Then, the K-value is scored using the WCSS objective function.

    Args:
        X: Feature matrix of shape (N, D)
        k: Number of clusters
        n_trials: Number of independent trials
        max_iters: Max iterations per run
        tol: Convergence tolerance
        random_state: Base random seed

    Returns:
        dict containing:
            - 'k': K-value
            - 'centroids': Final averaged centroids of shape (K, D)
            - 'assignments': Final cluster assignments for each point
            - 'score': Final objective function score J
            - 'trial_scores': List of objective scores from individual trials
            - 'trial_centroids': List of centroid matrices from all trials
            - 'inter_cluster_distances': Pairwise distance stats between centroids
    """
    X = np.asarray(X, dtype=np.float64)
    base_seed = random_state if random_state is not None else 42

    all_centroids = []
    trial_scores = []

    for trial_idx in range(n_trials):
        seed = base_seed + trial_idx * 1009
        km = KMeans(k=k, max_iters=max_iters, tol=tol, random_state=seed)
        km.fit(X)
        all_centroids.append(km.centroids)
        trial_scores.append(km.inertia_)

    # Align all trial centroids to the first trial using Hungarian matching
    ref_centroids = all_centroids[0]
    aligned_centroids = [ref_centroids]

    for c in all_centroids[1:]:
        aligned = align_centroids(ref_centroids, c)
        aligned_centroids.append(aligned)

    # Average final centroid positions across all trials
    averaged_centroids = np.mean(aligned_centroids, axis=0)

    # Compute final assignments and objective score using averaged centroids
    final_assignments = assign_clusters(X, averaged_centroids)
    final_score = objective_function(X, averaged_centroids, final_assignments)

    # Compute inter-cluster distances
    inter_cluster_dist = compute_inter_cluster_distances(averaged_centroids)

    return {
        "k": k,
        "centroids": averaged_centroids,
        "assignments": final_assignments,
        "score": final_score,
        "trial_scores": trial_scores,
        "trial_centroids": all_centroids,
        "inter_cluster_distances": inter_cluster_dist,
    }
