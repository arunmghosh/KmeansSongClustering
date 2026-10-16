"""
Data preprocessing module for Song Clustering using K-means.
Loads artist song data and prepares features without scaling
(features are already on the 0-7 scale for interpretability).
"""

from pathlib import Path
from typing import List, Union
import numpy as np
import pandas as pd

FEATURE_COLUMNS: List[str] = ["Energy", "Positivity", "Introspectiveness"]


def load_data(
    filepath: Union[str, Path] = "artist_data.xlsx",
    sheet_name: str = "artists_data",
) -> pd.DataFrame:
    """Load the song dataset from an Excel file and validate structure.

    Args:
        filepath: Path to artist_data.xlsx
        sheet_name: Sheet name containing song data

    Returns:
        pd.DataFrame: Cleaned DataFrame with original feature scales preserved.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path.resolve()}")

    df = pd.read_excel(path, sheet_name=sheet_name)

    expected_cols = ["Song", "Artist", "Album"] + FEATURE_COLUMNS
    missing_cols = [c for c in expected_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Dataset missing required columns: {missing_cols}")

    # Ensure numeric columns are properly typed
    for col in FEATURE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="raise")

    # Ensure no NaN values in features
    if df[FEATURE_COLUMNS].isnull().any().any():
        raise ValueError("Feature columns contain null/NaN values.")

    return df


def get_features(df: pd.DataFrame) -> np.ndarray:
    """Extract feature matrix for K-means clustering without scaling.

    Args:
        df: DataFrame containing FEATURE_COLUMNS

    Returns:
        np.ndarray: Matrix of shape (N, 3) as float64.
    """
    return df[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
