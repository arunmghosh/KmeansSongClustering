"""
Statistical analysis module for K-means song clustering.
Computes metrics for Stages 3 and 4 of results:
- Stage 3: Artist distribution across clusters and comparison to artist averages
- Stage 4: Artist versatility (Gini Impurity) and album trajectory shifts
"""

from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from preprocess import FEATURE_COLUMNS


def artist_cluster_distribution(
    df: pd.DataFrame, assignments: np.ndarray
) -> pd.DataFrame:
    """Compute the distribution of each artist's songs across clusters (Stage 3).

    Args:
        df: DataFrame containing 'Artist' column
        assignments: 1D array of cluster indices (length N)

    Returns:
        pd.DataFrame: Contingency table of shape (num_artists, num_clusters) with counts.
    """
    temp_df = df.copy()
    temp_df["Cluster"] = assignments

    crosstab = pd.crosstab(
        temp_df["Artist"], temp_df["Cluster"], margins=True, margins_name="Total"
    )
    return crosstab


def artist_feature_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the average Energy, Positivity, and Introspectiveness for each artist.

    Args:
        df: DataFrame with 'Artist' and FEATURE_COLUMNS

    Returns:
        pd.DataFrame: Mean feature values indexed by Artist.
    """
    return df.groupby("Artist")[FEATURE_COLUMNS].mean()


def compare_centroids_to_artist_averages(
    centroids: np.ndarray, artist_means: pd.DataFrame
) -> Dict[str, Any]:
    """Compare cluster centroids to the average feature profile of each artist (Stage 3).

    Args:
        centroids: Centroid matrix of shape (K, 3)
        artist_means: DataFrame indexed by Artist with columns [Energy, Positivity, Introspectiveness]

    Returns:
        dict containing:
            - 'distance_matrix': DataFrame of Euclidean distances (centroids x artists)
            - 'closest_artist_per_cluster': dict mapping cluster_idx -> closest artist
            - 'closest_cluster_per_artist': dict mapping artist -> closest cluster_idx
    """
    artists = artist_means.index.tolist()
    artist_vecs = artist_means.to_numpy(dtype=np.float64)  # (A, 3)

    # Compute Euclidean distance between each centroid (K, 3) and artist mean (A, 3)
    diff = centroids[:, np.newaxis, :] - artist_vecs[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=-1))  # (K, A)

    dist_df = pd.DataFrame(
        dists, index=[f"Cluster_{k}" for k in range(len(centroids))], columns=artists
    )

    closest_artist_per_cluster = {}
    for k_idx in range(len(centroids)):
        closest_a = artists[int(np.argmin(dists[k_idx]))]
        closest_artist_per_cluster[k_idx] = closest_a

    closest_cluster_per_artist = {}
    for a_idx, artist in enumerate(artists):
        closest_c = int(np.argmin(dists[:, a_idx]))
        closest_cluster_per_artist[artist] = closest_c

    return {
        "distance_matrix": dist_df,
        "closest_artist_per_cluster": closest_artist_per_cluster,
        "closest_cluster_per_artist": closest_cluster_per_artist,
    }


def compute_gini_impurity(
    assignments: Union[List[int], np.ndarray], k: Optional[int] = None
) -> float:
    """Compute the Gini Impurity metric for a set of cluster assignments:

    Gini = 1 - sum_{i=1}^k (p_i)^2
    where p_i is the proportion of songs assigned to cluster i.

    - 0.0 indicates complete purity (all songs in a single cluster, least versatile).
    - Higher values indicate songs are scattered across multiple clusters (higher versatility).

    Args:
        assignments: Cluster labels for an artist's songs
        k: Optional total number of clusters (for validation)

    Returns:
        float: Gini impurity in range [0, 1 - 1/k].
    """
    arr = np.asarray(assignments)
    n = len(arr)
    if n == 0:
        return 0.0

    _, counts = np.unique(arr, return_counts=True)
    probabilities = counts / n
    gini = 1.0 - float(np.sum(probabilities**2))
    return float(max(0.0, gini))


def compute_versatility_scores(
    df: pd.DataFrame, assignments: np.ndarray, k: Optional[int] = None
) -> pd.DataFrame:
    """Compute versatility scores for each artist based on the Gini Impurity

    of their song cluster assignments (Stage 4).

    Args:
        df: DataFrame containing 'Artist'
        assignments: 1D array of cluster indices (length N)
        k: Total number of clusters K

    Returns:
        pd.DataFrame: Table with Artist, Song_Count, Clusters_Touched,
                      Gini_Impurity, and Normalized_Gini (sorted descending by versatility).
    """
    temp_df = df.copy()
    temp_df["Cluster"] = assignments

    total_k = k if k is not None else len(np.unique(assignments))
    max_possible_gini = (1.0 - 1.0 / total_k) if total_k > 1 else 1.0

    records = []
    for artist, group in temp_df.groupby("Artist"):
        artist_clusters = group["Cluster"].to_numpy()
        gini = compute_gini_impurity(artist_clusters, k=total_k)
        norm_gini = gini / max_possible_gini if max_possible_gini > 0 else 0.0
        unique_clusters = len(np.unique(artist_clusters))

        records.append(
            {
                "Artist": artist,
                "Song_Count": len(group),
                "Clusters_Touched": unique_clusters,
                "Gini_Impurity": round(gini, 4),
                "Normalized_Gini": round(norm_gini, 4),
            }
        )

    result_df = pd.DataFrame(records).sort_values(
        by=["Gini_Impurity", "Clusters_Touched"], ascending=[False, False]
    )
    return result_df.reset_index(drop=True)


def album_centers(df: pd.DataFrame) -> pd.DataFrame:
    """Compute the mean feature coordinates for each album of each artist (Stage 4)."""
    return df.groupby(["Artist", "Album"])[FEATURE_COLUMNS].mean()


def album_shifts(df: pd.DataFrame) -> pd.DataFrame:
    """Compare centers of each artist's first and last album and calculate

    the Euclidean distance shift in feature space (Stage 4).

    Args:
        df: DataFrame containing 'Artist', 'Album', and FEATURE_COLUMNS

    Returns:
        pd.DataFrame: Album pairs and distance shift for each artist.
    """
    records = []
    for artist, group in df.groupby("Artist"):
        albums = group["Album"].unique().tolist()
        if len(albums) >= 2:
            alb1, alb2 = albums[0], albums[1]
            c1 = group[group["Album"] == alb1][FEATURE_COLUMNS].mean().to_numpy()
            c2 = group[group["Album"] == alb2][FEATURE_COLUMNS].mean().to_numpy()
            shift_dist = float(np.sqrt(np.sum((c1 - c2) ** 2)))

            records.append(
                {
                    "Artist": artist,
                    "Album_1": alb1,
                    "Album_2": alb2,
                    "Shift_Distance": round(shift_dist, 4),
                    "Center_1": np.round(c1, 3).tolist(),
                    "Center_2": np.round(c2, 3).tolist(),
                }
            )
        elif len(albums) == 1:
            records.append(
                {
                    "Artist": artist,
                    "Album_1": albums[0],
                    "Album_2": albums[0],
                    "Shift_Distance": 0.0,
                    "Center_1": np.round(group[FEATURE_COLUMNS].mean().to_numpy(), 3).tolist(),
                    "Center_2": np.round(group[FEATURE_COLUMNS].mean().to_numpy(), 3).tolist(),
                }
            )

    result_df = pd.DataFrame(records).sort_values(
        by="Shift_Distance", ascending=False
    )
    return result_df.reset_index(drop=True)


def evaluate_cluster_classification(
    y_true: Union[List[str], np.ndarray, pd.Series], cluster_assignments: np.ndarray
) -> Dict[str, Any]:
    """Evaluate cluster assignments for classification (Phase 1 & Phase 2).

    Assigns each cluster to the majority ground-truth class, then computes
    overall classification accuracy and cluster purity.

    Args:
        y_true: True class labels (e.g. artist or album)
        cluster_assignments: 1D cluster indices

    Returns:
        dict containing:
            - 'accuracy': Overall classification accuracy
            - 'cluster_to_class': dict mapping cluster id -> majority class label
            - 'purity': Cluster purity
            - 'contingency_matrix': Contingency table of true classes vs clusters
    """
    y_arr = np.asarray(y_true)
    c_arr = np.asarray(cluster_assignments)
    n = len(y_arr)

    if n == 0:
        return {"accuracy": 0.0, "cluster_to_class": {}, "purity": 0.0}

    unique_clusters = np.unique(c_arr)
    cluster_to_class = {}
    correct = 0

    for c in unique_clusters:
        mask = c_arr == c
        classes_in_c, counts = np.unique(y_arr[mask], return_counts=True)
        majority_class = classes_in_c[np.argmax(counts)]
        cluster_to_class[int(c)] = majority_class
        correct += int(np.max(counts))

    accuracy = float(correct / n)
    purity = accuracy

    contingency = pd.crosstab(
        pd.Series(y_arr, name="True_Class"),
        pd.Series(c_arr, name="Cluster"),
    )

    return {
        "accuracy": round(accuracy, 4),
        "cluster_to_class": cluster_to_class,
        "purity": round(purity, 4),
        "contingency_matrix": contingency,
    }
