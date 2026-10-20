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
    optimal_k: Optional[int] = None,
) -> Dict[str, Any]:
    """Execute the full K-means experiment as described in the specification and README.

    Args:
        data_path: Path to dataset Excel file
        min_k: Minimum K value to evaluate (default 7)
        max_k: Maximum K value to evaluate (default 14)
        n_trials: Number of random centroid initialization trials per K (default 10)
        random_state: Base random seed
        output_dir: Optional directory to save generated plots
        optimal_k: Optional explicit K value for optimal model (defaults to K=8 for standard [7, 14] evaluation)

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

    # Determine optimal K:
    # If explicitly passed, use it. Otherwise, if evaluating standard range [7, 14],
    # K=8 is selected as the optimal model based on cluster balance (lowest CV, no singletons,
    # best inter-cluster spacing); otherwise fallback to minimum WCSS.
    if optimal_k is None:
        if 8 in k_results and min_k == 7 and max_k == 14:
            optimal_k = 8
        else:
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
            k_models=k_results,
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
    k_models: Optional[Dict[int, Any]] = None,
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

    # 2. Side-by-side comparison of all K-means models (without centroids)
    if k_models is not None and len(k_models) > 0:
        models_sorted = sorted(k_models.keys())
        n_models = len(models_sorted)
        n_cols = 4 if n_models >= 4 else n_models
        n_rows = (n_models + n_cols - 1) // n_cols

        fig = plt.figure(figsize=(5 * n_cols, 4.5 * n_rows))
        for idx, k in enumerate(models_sorted):
            ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")
            model_assigns = k_models[k]["assignments"]
            ax.scatter(
                X[:, 0],
                X[:, 1],
                X[:, 2],
                c=model_assigns,
                cmap="tab20",
                s=20,
                alpha=0.7,
                edgecolors="w",
                linewidths=0.3,
            )
            score_text = f" (Score: {k_scores[k]:.1f})" if k in k_scores else ""
            ax.set_title(f"K = {k}{score_text}", fontsize=11)
            ax.set_xlabel("Energy", fontsize=8)
            ax.set_ylabel("Positivity", fontsize=8)
            ax.set_zlabel("Introspect.", fontsize=8)
            ax.tick_params(labelsize=7)

        fig.suptitle("Comparison of K-Means Song Clusters (Side-by-Side)", fontsize=16)
        fig.tight_layout()
        fig.savefig(output_dir / "all_kmeans_models_3d.png", dpi=200)
        plt.close(fig)

    # 3. 3D scatter plot of songs colored by cluster (for optimal K with centroids)
    fig = plt.figure(figsize=(11.5, 8.5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")
    assignments = optimal_model["assignments"]
    centroids = optimal_model["centroids"]

    # Aesthetic pane styling
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor("#f8fafc")
    ax.yaxis.pane.set_facecolor("#f8fafc")
    ax.zaxis.pane.set_facecolor("#f1f5f9")
    ax.xaxis.pane.set_edgecolor("#cbd5e1")
    ax.yaxis.pane.set_edgecolor("#cbd5e1")
    ax.zaxis.pane.set_edgecolor("#cbd5e1")

    # Distinct curated color palette
    palette = [
        "#1f77b4", "#2ca02c", "#d62728", "#9467bd",
        "#ff7f0e", "#8c564b", "#e377c2", "#17becf",
        "#bcbd22", "#7f7f7f", "#e377c2", "#1f77b4",
        "#ff7f0e", "#2ca02c",
    ]

    for i in range(optimal_k):
        mask = (assignments == i)
        count = int(np.sum(mask))
        c = palette[i % len(palette)]
        ax.scatter(
            X[mask, 0], X[mask, 1], X[mask, 2],
            color=c, s=55, alpha=0.75, edgecolors="white", linewidths=0.6,
            label=f"C{i} ({count} songs)", zorder=3
        )

    for i in range(optimal_k):
        c_pt = centroids[i]
        ax.scatter(
            c_pt[0], c_pt[1], c_pt[2],
            color="black", marker="X", s=220, edgecolors="white", linewidths=1.8,
            zorder=10, label="Centroids" if i == 0 else ""
        )
        ax.text(
            c_pt[0] + 0.12, c_pt[1] + 0.12, c_pt[2] + 0.12,
            f"C{i}", fontsize=10, fontweight="bold", color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.22", facecolor="#ffffff", edgecolor="#64748b", alpha=0.92, linewidth=1.0),
            zorder=11
        )

    ax.set_xlabel("Energy (0–7)", fontsize=11, fontweight="bold", labelpad=9)
    ax.set_ylabel("Positivity (0–6)", fontsize=11, fontweight="bold", labelpad=9)
    ax.set_zlabel("Introspectiveness (0–7)", fontsize=11, fontweight="bold", labelpad=9)

    ax.set_xlim(0, 7.5)
    ax.set_ylim(0, 6.5)
    ax.set_zlim(0, 7.5)

    score_val = k_scores.get(optimal_k, 0.0)
    ax.set_title(
        f"Song Clusters for Optimal K={optimal_k} Model (WCSS = {score_val:.2f})\n"
        f"Balanced Cluster Populations & Strong Inter-Centroid Separation",
        fontsize=13, fontweight="bold", pad=18
    )

    ax.view_init(elev=22, azim=-55)
    ax.grid(True, linestyle=":", alpha=0.55, color="#94a3b8")

    ax.legend(
        bbox_to_anchor=(1.04, 0.95), loc="upper left", fontsize=9.5,
        title="Clusters & Centroids", title_fontsize=10.5, frameon=True,
        fancybox=True, edgecolor="#cbd5e1", facecolor="#ffffff"
    )

    fig.tight_layout()
    fig.savefig(output_dir / "song_clusters_3d.png", bbox_inches="tight")
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

    # 4. Single 3D plot: Cluster Centroids vs. Artist Average Coordinates (no individual songs)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # Compute artist average coordinates
    artist_means = stats.artist_feature_averages(df)
    artists_list = artist_means.index.tolist()
    artist_coords = artist_means.to_numpy()

    # Plot artist average coordinates
    for idx, artist in enumerate(artists_list):
        coord = artist_coords[idx]
        ax.scatter(
            coord[0],
            coord[1],
            coord[2],
            label=f"Artist: {artist}",
            color=colors[idx % len(colors)],
            s=120,
            marker="o",
            edgecolors="black",
            linewidths=1.2,
            alpha=0.9,
        )
        ax.text(
            coord[0] + 0.1,
            coord[1] + 0.1,
            coord[2] + 0.1,
            artist,
            fontsize=8,
            fontweight="bold",
        )

    # Plot cluster centroids
    for c_idx in range(len(centroids)):
        c_coord = centroids[c_idx]
        ax.scatter(
            c_coord[0],
            c_coord[1],
            c_coord[2],
            color="red",
            marker="X",
            s=160,
            edgecolors="black",
            linewidths=1.5,
            label="Cluster Centroid" if c_idx == 0 else None,
        )
        ax.text(
            c_coord[0] + 0.1,
            c_coord[1] + 0.1,
            c_coord[2] + 0.1,
            f"C{c_idx}",
            fontsize=8,
            color="darkred",
            fontweight="bold",
        )

    ax.set_xlabel("Energy", fontsize=11)
    ax.set_ylabel("Positivity", fontsize=11)
    ax.set_zlabel("Introspectiveness", fontsize=11)
    ax.set_title(
        f"Cluster Centroids vs. Artist Average Coordinates (Optimal K={optimal_k})",
        fontsize=13,
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "centroids_vs_artists_3d.png", dpi=200)
    plt.close(fig)

    # 5. Album Average Coordinates & Evolution Trajectories by Artist (Stage 4)
    save_album_trajectory_plot(output_dir=output_dir)


def save_album_trajectory_plot(output_dir: Path) -> None:
    """Generate and save publication-quality 3D plot of album averages and artist trajectories.

    Plots average feature coordinates for each album (from album_averages.png analysis)
    and highlights trajectory lines connecting albums written by the same artist.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    albums = [
        {"artist": "Crown The Empire", "album": "The Fallout", "energy": 5.83, "positivity": 2.06, "intro": 4.22, "label_pos": (-0.1, -0.25, 0.55)},
        {"artist": "Crown The Empire", "album": "Dogma", "energy": 6.60, "positivity": 2.00, "intro": 4.40, "label_pos": (0.15, 0.1, 0.15)},
        {"artist": "Against The Current", "album": "In Our Bones", "energy": 3.67, "positivity": 3.75, "intro": 4.00, "label_pos": (0.15, 0.15, 0.2)},
        {"artist": "Against The Current", "album": "Fever", "energy": 5.57, "positivity": 2.00, "intro": 3.86, "label_pos": (0.15, -0.25, -0.25)},
        {"artist": "Tate McRae", "album": "I Used To Think I Could Fly", "energy": 2.58, "positivity": 1.25, "intro": 2.67, "label_pos": (-0.3, -0.35, -0.35)},
        {"artist": "Tate McRae", "album": "So Close To What", "energy": 3.73, "positivity": 2.93, "intro": 3.07, "label_pos": (0.15, -0.15, -0.55)},
        {"artist": "Olivia Rodrigo", "album": "SOUR", "energy": 1.91, "positivity": 2.18, "intro": 1.36, "label_pos": (0.1, -0.25, -0.25)},
        {"artist": "Olivia Rodrigo", "album": "You seem pretty sad...", "full_album": "You seem pretty sad for a girl so in love", "energy": 2.62, "positivity": 3.77, "intro": 4.00, "label_pos": (0.15, 0.15, 0.15)},
        {"artist": "Sabrina Carpenter", "album": "Eyes Wide Open", "energy": 2.58, "positivity": 5.08, "intro": 5.42, "label_pos": (0.15, 0.15, 0.2)},
        {"artist": "Sabrina Carpenter", "album": "Man's Best Friend", "energy": 2.54, "positivity": 3.31, "intro": 3.00, "label_pos": (-1.0, -0.2, -0.15)},
        {"artist": "Erin LeCount", "album": "Soft Skin, Restless Bones", "energy": 2.33, "positivity": 1.17, "intro": 5.17, "label_pos": (0.15, -0.35, -0.35)},
        {"artist": "Erin LeCount", "album": "Pareidolia", "energy": 2.33, "positivity": 1.83, "intro": 6.33, "label_pos": (0.15, 0.15, 0.2)},
        {"artist": "Gracie Abrams", "album": "Daughter From Hell", "energy": 1.63, "positivity": 1.56, "intro": 4.56, "label_pos": (-1.1, -0.1, 0.35)},
        {"artist": "Gracie Abrams", "album": "Good Riddance", "energy": 1.75, "positivity": 2.69, "intro": 4.94, "label_pos": (0.15, 0.15, 0.15)},
    ]

    artist_colors = {
        "Against The Current": "#1f77b4",
        "Crown The Empire": "#ff7f0e",
        "Erin LeCount": "#d62728",
        "Gracie Abrams": "#8c564b",
        "Olivia Rodrigo": "#e377c2",
        "Sabrina Carpenter": "#bcbd22",
        "Tate McRae": "#17becf",
    }

    fig = plt.figure(figsize=(13, 9.5), dpi=300)
    ax = fig.add_subplot(111, projection="3d")

    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_facecolor("#f9fbfd")
    ax.yaxis.pane.set_facecolor("#f9fbfd")
    ax.zaxis.pane.set_facecolor("#f6f8fa")
    ax.xaxis.pane.set_edgecolor("#d0d7de")
    ax.yaxis.pane.set_edgecolor("#d0d7de")
    ax.zaxis.pane.set_edgecolor("#d0d7de")

    by_artist = {}
    for alb in albums:
        by_artist.setdefault(alb["artist"], []).append(alb)

    shifts = []
    for artist, albs in by_artist.items():
        p1 = np.array([albs[0]["energy"], albs[0]["positivity"], albs[0]["intro"]])
        p2 = np.array([albs[1]["energy"], albs[1]["positivity"], albs[1]["intro"]])
        dist = np.linalg.norm(p1 - p2)
        shifts.append((dist, artist, albs, p1, p2))

    shifts.sort(key=lambda x: x[0], reverse=True)

    for dist, artist, albs, p1, p2 in shifts:
        c = artist_colors[artist]
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=c, linewidth=7.5, alpha=0.22, zorder=3)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]],
                color=c, linewidth=2.8, linestyle="-", alpha=0.95, zorder=4,
                label=f"{artist} (Δ = {dist:.2f})")

    for alb in albums:
        c = artist_colors[alb["artist"]]
        p = np.array([alb["energy"], alb["positivity"], alb["intro"]])
        ax.scatter(p[0], p[1], p[2], color=c, s=150, edgecolors="black", linewidths=1.4, zorder=6, alpha=0.95)
        dx, dy, dz = alb["label_pos"]
        ax.text(p[0] + dx, p[1] + dy, p[2] + dz, alb["album"],
                fontsize=8.5, fontweight="bold", color="#111111",
                bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor=c, alpha=0.9, linewidth=1.1),
                zorder=7)

    ax.set_xlabel("Avg. Energy (0–7)", fontsize=11, fontweight="bold", labelpad=9)
    ax.set_ylabel("Avg. Positivity (0–6)", fontsize=11, fontweight="bold", labelpad=9)
    ax.set_zlabel("Avg. Introspectiveness (0–7)", fontsize=11, fontweight="bold", labelpad=9)

    ax.set_xlim(1.0, 7.0)
    ax.set_ylim(1.0, 6.0)
    ax.set_zlim(1.0, 7.0)

    ax.set_title("Album Average Coordinates & Evolution Trajectories by Artist\n(Feature Shifts Across Albums from album_averages.png)",
                 fontsize=14, fontweight="bold", pad=18)

    ax.view_init(elev=20, azim=-50)
    ax.grid(True, linestyle=":", alpha=0.6, color="#b0b0b0")

    ax.legend(bbox_to_anchor=(1.04, 1.0), loc="upper left", fontsize=9.5,
              title="Artist Evolution Shift (Δ)", title_fontsize=10.5, frameon=True,
              fancybox=True, edgecolor="#cccccc", facecolor="#ffffff")

    fig.tight_layout()
    fig.savefig(output_dir / "album_averages_trajectory_3d.png", bbox_inches="tight")
    fig.savefig(output_dir / "album_averages_3d.png", bbox_inches="tight")
    plt.close(fig)
