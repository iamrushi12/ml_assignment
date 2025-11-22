"""
Feature engineering for fuel price optimization.

This module builds features that capture competition, seasonality,
and short-term demand patterns via lag and rolling-window statistics.
"""

import pandas as pd

# Feature and target names used across training and inference.
FEATURE_COLUMNS = [
    "price",
    "cost",
    "comp1_price",
    "comp2_price",
    "comp3_price",
    "avg_comp_price",
    "price_gap_vs_avg",
    "day_of_week",
    "month",
    "lag1_volume",
    "lag7_volume",
    "rolling_7d_vol_mean",
    "rolling_7d_price_mean",
    "trend_index",
]

TARGET_COLUMN = "volume"


def add_competitor_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add average competitor price and price gap vs average."""
    comp_cols = ["comp1_price", "comp2_price", "comp3_price"]
    df["avg_comp_price"] = df[comp_cols].mean(axis=1)
    df["price_gap_vs_avg"] = df["price"] - df["avg_comp_price"]
    return df


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add calendar features (day of week, month)."""
    df["day_of_week"] = df["date"].dt.dayofweek
    df["month"] = df["date"].dt.month
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add lag and rolling-window volume/price features.
    Assumes df['date'] is datetime and df is sorted by date.
    """
    df = df.sort_values("date")
    df["lag1_volume"] = df["volume"].shift(1)
    df["lag7_volume"] = df["volume"].shift(7)
    df["rolling_7d_vol_mean"] = df["volume"].rolling(7).mean()
    df["rolling_7d_price_mean"] = df["price"].rolling(7).mean()
    return df


def add_trend_feature(df: pd.DataFrame) -> pd.DataFrame:
    """Add a simple time index from the first date."""
    df["trend_index"] = (df["date"] - df["date"].min()).dt.days
    return df


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline for historical data.

    Returns a DataFrame that still contains:
      - original 'date' and 'volume' columns
      - all engineered feature columns
    """
    df = add_competitor_features(df)
    df = add_calendar_features(df)
    df = add_lag_features(df)
    df = add_trend_feature(df)

    # Drop rows where lag/rolling features are not available
    df = df.dropna().reset_index(drop=True)
    return df
