"""
Experiment pipeline orchestrator for K-means song clustering.
Constructs the full experiment stages:
1. Optimal K determination across K-values
2. Distribution of clusters (inter-cluster distance analysis)
3. Distribution of artists' songs and comparison to artist averages
4. Versatility analysis (Gini Impurity) and album progression shifts
5. Individual phase executions (Phase 1 binary, Phase 2 multinomial)
6. Smoke testing interface.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
import pandas as pd

import kmeans
import preprocess
import stats


def run_experiment(
    data_path: Union[str, Path] = "artist_data.xlsx",
    min_k: int = 7,
    max_k: int = 14,
    n_trials: int = 10,
    random_state: int = 42,
    output_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Execute the full K-means experiment as described in the specification and README.

    Args:
        data_path: Path to dataset Excel file
        min_k: Minimum K value to evaluate (default 7)
        max_k: Maximum K value to evaluate (default 14)
        n_trials: Number of random centroid initialization trials per K (default 10)
        random_state: Base random seed
        output_dir: Optional directory to save generated plots

    Returns:
        dict: Complete structured experiment results.
    """
    df = preprocess.load_data(data_path)
    X = preprocess.get_features(df)

    k_results = {}
    k_scores = {}

    for k in range(min_k, max_k + 1):
        res = kmeans.run_kmeans_trials(
            X, k=k, n_trials=n_trials, random_state=random_state
        )
        k_results[k] = res
        k_scores[k] = res["score"]

    # Choose optimal K with the smallest objective function score
    optimal_k = min(k_scores, key=k_scores.get)
    optimal_model = k_results[optimal_k]

    # Stage 2: Cluster distribution & inter-cluster distances
    inter_cluster_summary = {
        k: {
            "min_distance": k_results[k]["inter_cluster_distances"]["min_distance"],
            "mean_distance": k_results[k]["inter_cluster_distances"]["mean_distance"],
            "max_distance": k_results[k]["inter_cluster_distances"]["max_distance"],
        }
        for k in k_results
    }

    # Stage 3: Distribution of artists' songs & artist averages
    artist_crosstab = stats.artist_cluster_distribution(
        df, optimal_model["assignments"]
    )
    artist_means = stats.artist_feature_averages(df)
    centroid_comparison = stats.compare_centroids_to_artist_averages(
        optimal_model["centroids"], artist_means
    )

    # Stage 4: Versatility (Gini Impurity) & Album trajectory shifts
    versatility_df = stats.compute_versatility_scores(
        df, optimal_model["assignments"], k=optimal_k
    )
    album_shifts_df = stats.album_shifts(df)

    # Optional plot generation
    if output_dir is not None:
        save_experiment_plots(
            df=df,
            X=X,
            k_scores=k_scores,
            optimal_k=optimal_k,
            optimal_model=optimal_model,
            output_dir=Path(output_dir),
        )

    return {
        "dataset_shape": df.shape,
        "k_range": list(range(min_k, max_k + 1)),
        "k_scores": k_scores,
        "optimal_k": optimal_k,
        "optimal_score": k_scores[optimal_k],
        "k_models": k_results,
        "stage2_inter_cluster_distances": inter_cluster_summary,
        "stage3_artist_crosstab": artist_crosstab,
        "stage3_artist_means": artist_means,
        "stage3_centroid_comparison": centroid_comparison,
        "stage4_versatility": versatility_df,
        "stage4_album_shifts": album_shifts_df,
    }


def run_smoke_test(
    data_path: Union[str, Path] = "artist_data.xlsx", random_state: int = 42
) -> bool:
    """Run a rapid end-to-end integration check across the experiment pipeline.

    Args:
        data_path: Path to dataset
        random_state: Random seed

    Returns:
        bool: True if all checks pass.
    """
    full_res = run_experiment(
        data_path=data_path,
        min_k=2,
        max_k=3,
        n_trials=2,
        random_state=random_state,
    )
    assert "optimal_k" in full_res
    assert full_res["optimal_k"] in [2, 3]
    assert len(full_res["stage4_versatility"]) == 7
    assert len(full_res["stage4_album_shifts"]) == 7
    assert "stage3_artist_crosstab" in full_res
    assert "stage2_inter_cluster_distances" in full_res
    return True


def save_experiment_plots(
    df: pd.DataFrame,
    X: np.ndarray,
    k_scores: Dict[int, float],
    optimal_k: int,
    optimal_model: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate and save publication-quality figures for the experiment."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Elbow curve (K vs Objective Score)
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = sorted(k_scores.keys())
    scores = [k_scores[k] for k in ks]
    ax.plot(ks, scores, "bo-", linewidth=2, markersize=8)
    ax.axvline(optimal_k, color="red", linestyle="--", label=f"Optimal K={optimal_k}")
    ax.set_title("K-means Objective Function Score vs K", fontsize=14)
    ax.set_xlabel("Number of Clusters (K)", fontsize=12)
    ax.set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=12)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "elbow_curve.png", dpi=200)
    plt.close(fig)

    # 2. 3D scatter plot of songs colored by cluster (for optimal K)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    assignments = optimal_model["assignments"]
    centroids = optimal_model["centroids"]

    scatter = ax.scatter(
        X[:, 0],
        X[:, 1],
        X[:, 2],
        c=assignments,
        cmap="tab10",
        s=40,
        alpha=0.7,
        edgecolors="w",
    )
    ax.scatter(
        centroids[:, 0],
        centroids[:, 1],
        centroids[:, 2],
        c="red",
        marker="X",
        s=150,
        edgecolors="black",
        linewidths=1.5,
        label="Centroids",
    )
    ax.set_xlabel("Energy")
    ax.set_ylabel("Positivity")
    ax.set_zlabel("Introspectiveness")
    ax.set_title(f"Song Clusters for Optimal K={optimal_k}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "song_clusters_3d.png", dpi=200)
    plt.close(fig)

    # 3. 3D scatter plot of songs colored by artist
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    artists = df["Artist"].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(artists)))

    for idx, artist in enumerate(artists):
        mask = (df["Artist"] == artist).to_numpy()
        ax.scatter(
            X[mask, 0],
            X[mask, 1],
            X[mask, 2],
            label=artist,
            color=colors[idx],
            s=40,
            alpha=0.8,
            edgecolors="w",
        )

    ax.set_xlabel("Energy")
    ax.set_ylabel("Positivity")
    ax.set_zlabel("Introspectiveness")
    ax.set_title("Songs by Artist (3D Feature Space)")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(output_dir / "songs_by_artist_3d.png", dpi=200)
    plt.close(fig)
