"""
Data ingestion and basic cleaning for oil retail history.
"""

from typing import Optional

import pandas as pd

from .config import DATA_RAW_HISTORY_PATH


def load_raw_history(path: Optional[str] = None) -> pd.DataFrame:
    """
    Load raw daily fuel history from CSV.

    Expects at least the following columns:
        - date
        - price
        - cost
        - comp1_price, comp2_price, comp3_price
        - volume
    """
    if path is None:
        path = DATA_RAW_HISTORY_PATH

    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values("date").reset_index(drop=True)

    expected_cols = {
        "date",
        "price",
        "cost",
        "comp1_price",
        "comp2_price",
        "comp3_price",
        "volume",
    }
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in history CSV: {missing}")

    return df


def clean_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Basic cleaning and validation:
      - drop rows with missing critical fields
      - remove non-positive prices or costs
      - remove negative volumes
    """
    # Drop rows with missing critical values
    df = df.dropna(subset=["date", "price", "cost", "volume"])

    # Only keep rows with valid economic ranges
    df = df[df["price"] > 0]
    df = df[df["cost"] > 0]
    df = df[df["volume"] >= 0]

    # (Optional) more domain-specific checks could be added here

    return df.reset_index(drop=True)
