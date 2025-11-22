"""
Pricing engine: uses the trained volume model to recommend a daily price
that maximizes predicted profit under business constraints.
"""

from typing import Any, Dict, List, Tuple
import json
import pickle

import numpy as np
import pandas as pd

from .config import (
    MAX_ABS_PRICE_CHANGE,
    MIN_MARGIN_PER_LITER,
    COMPETITIVE_MAX_DELTA,
    PRICE_GRID_STEP,
    MIN_PRICE,
    MAX_PRICE,
    MODEL_PATH,
    FEATURE_CONFIG_PATH,
)
from .features import FEATURE_COLUMNS


def load_model_and_config() -> Tuple[Any, List[str]]:
    """
    Load the trained model and feature configuration from disk.
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURE_CONFIG_PATH, "r") as f:
        cfg = json.load(f)

    feature_cols = cfg.get("feature_columns", FEATURE_COLUMNS)
    return model, feature_cols


def build_price_grid(today_cost: float, last_price: float, avg_comp: float) -> np.ndarray:
    """
    Build a candidate price grid constrained by:
      - overall min/max allowed price
      - max absolute change vs last company price
      - minimum margin vs cost
      - maximum allowed gap above avg competitor price
    """
    low = max(
        MIN_PRICE,
        today_cost + MIN_MARGIN_PER_LITER,
        last_price - MAX_ABS_PRICE_CHANGE,
    )

    high = min(
        MAX_PRICE,
        last_price + MAX_ABS_PRICE_CHANGE,
        avg_comp + COMPETITIVE_MAX_DELTA,
    )

    if low >= high:
        # Degenerate case: collapse to a single point and extend slightly
        high = low + PRICE_GRID_STEP

    prices = np.arange(low, high + PRICE_GRID_STEP, PRICE_GRID_STEP)
    return np.round(prices, 2)


def build_feature_row_for_candidate(
    today: Dict, candidate_price: float, history_features: pd.DataFrame
) -> Dict:
    """
    Build a feature-row for a single candidate price on 'today'.

    We reuse the latest historical lag/rolling statistics and adjust
    the features that depend on today's price and date.
    """
    history_sorted = history_features.sort_values("date").reset_index(drop=True)
    last_row = history_sorted.iloc[-1]

    date_today = pd.to_datetime(today["date"])
    date_min = history_sorted["date"].min()

    avg_comp_price = float(
        np.mean(
            [
                today["comp1_price"],
                today["comp2_price"],
                today["comp3_price"],
            ]
        )
    )

    feature_row: Dict[str, float] = {
        "price": float(candidate_price),
        "cost": float(today["cost"]),
        "comp1_price": float(today["comp1_price"]),
        "comp2_price": float(today["comp2_price"]),
        "comp3_price": float(today["comp3_price"]),
        "avg_comp_price": avg_comp_price,
        "price_gap_vs_avg": float(candidate_price) - avg_comp_price,
        "day_of_week": float(date_today.dayofweek),
        "month": float(date_today.month),
        # For lag/rolling features we reuse latest available stats.
        "lag1_volume": float(last_row["volume"]),
        "lag7_volume": float(last_row["lag7_volume"]),
        "rolling_7d_vol_mean": float(last_row["rolling_7d_vol_mean"]),
        "rolling_7d_price_mean": float(last_row["rolling_7d_price_mean"]),
        "trend_index": float((date_today - date_min).days),
    }

    return feature_row


def recommend_price_for_today(
    today: Dict,
    history_features: pd.DataFrame,
    model: Any,
    feature_cols: List[str],
) -> Dict:
    """
    Recommend the best price for 'today' using a preloaded model
    and historical feature table.
    """
    last_price = float(today["price"])
    today_cost = float(today["cost"])

    avg_comp = float(
        np.mean(
            [
                today["comp1_price"],
                today["comp2_price"],
                today["comp3_price"],
            ]
        )
    )

    prices = build_price_grid(today_cost, last_price, avg_comp)

    candidates: List[Dict] = []

    for p in prices:
        feat_row = build_feature_row_for_candidate(today, float(p), history_features)
        X = pd.DataFrame([feat_row])[feature_cols]
        pred_volume = float(model.predict(X)[0])
        pred_profit = (float(p) - today_cost) * pred_volume

        candidates.append(
            {
                "price": float(p),
                "predicted_volume": pred_volume,
                "predicted_profit": pred_profit,
            }
        )

    if not candidates:
        raise RuntimeError("No candidate prices generated for today.")

    best = max(candidates, key=lambda x: x["predicted_profit"])

    return {
        "recommended_price": best["price"],
        "expected_volume": best["predicted_volume"],
        "expected_profit": best["predicted_profit"],
        "num_candidates_evaluated": len(candidates),
    }


def recommend_price(today: Dict, history_features: pd.DataFrame) -> Dict:
    """
    Convenience wrapper: loads model from disk and computes recommendation.

    Useful for CLI/demo scripts. For high-throughput serving, prefer
    preloading the model and using `recommend_price_for_today`.
    """
    model, feature_cols = load_model_and_config()
    return recommend_price_for_today(today, history_features, model, feature_cols)
