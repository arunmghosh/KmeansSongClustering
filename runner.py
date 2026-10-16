"""
Command-line interface for Song Clustering using K-means.
Supports custom trials, rapid smoke tests, and full experiment run.

Usage:
    python3 runner.py
    python3 runner.py --smoke-test
    python3 runner.py --trials 25 --minK 7 --maxK 14
    python3 runner.py --output ./plots
"""

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import experiment


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyzing Artist Styles Using K-means Clustering"
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        default=False,
        help="Run a rapid end-to-end integration check",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=10,
        help="Number of replications of each K-means model (default: 10)",
    )
    parser.add_argument(
        "--minK",
        type=int,
        default=7,
        help="Smallest K-value for K-means (default: 7)",
    )
    parser.add_argument(
        "--maxK",
        type=int,
        default=14,
        help="Largest K-value for K-means (default: 14)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to export plots (default: None)",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="artist_data.xlsx",
        help="Path to dataset Excel file (default: artist_data.xlsx)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Main CLI entrypoint."""
    parsed = parse_args(args)

    if parsed.smoke_test:
        print("=" * 60)
        print("RUNNING RAPID SMOKE TEST (End-to-End Pipeline Integration Check)")
        print("=" * 60)
        success = experiment.run_smoke_test(
            data_path=parsed.data_path, random_state=parsed.seed
        )
        if success:
            print("[SMOKE TEST PASSED] All pipeline stages executed successfully.")
            return 0
        else:
            print("[SMOKE TEST FAILED] Pipeline smoke test failed.")
            return 1

    # Run full experiment
    print("=" * 60)
    print(f"RUNNING FULL EXPERIMENT (K={parsed.minK}..{parsed.maxK}, trials={parsed.trials})")
    print("=" * 60)
    res = experiment.run_experiment(
        data_path=parsed.data_path,
        min_k=parsed.minK,
        max_k=parsed.maxK,
        n_trials=parsed.trials,
        random_state=parsed.seed,
        output_dir=parsed.output,
    )

    print(f"Dataset shape: {res['dataset_shape']}")
    print(f"Optimal K: {res['optimal_k']} (WCSS Objective Score: {res['optimal_score']:.2f})")
    print("\nObjective Scores across K:")
    for k in res["k_range"]:
        print(f"  K={k:2d}: WCSS = {res['k_scores'][k]:.2f}")

    print("\nArtist Versatility Rankings (Gini Impurity):")
    print(res["stage4_versatility"].to_string(index=False))

    print("\nAlbum Trajectory Shifts:")
    print(res["stage4_album_shifts"][["Artist", "Album_1", "Album_2", "Shift_Distance"]].to_string(index=False))

    if parsed.output:
        print(f"\nPlots saved to {parsed.output}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
