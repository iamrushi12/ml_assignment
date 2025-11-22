"""
FastAPI app exposing the pricing engine as an HTTP API.
"""

from typing import Any, List

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ..data_pipeline import load_raw_history, clean_history
from ..features import build_feature_table
from ..pricing import load_model_and_config, recommend_price_for_today

app = FastAPI(title="Fuel Price Optimizer", version="1.0.0")

# Globals initialized at startup
_history_features: pd.DataFrame | None = None
_model: Any | None = None
_feature_cols: List[str] | None = None


class TodayRequest(BaseModel):
    date: str = Field(..., description="Date for which price is to be recommended")
    price: float = Field(..., description="Last observed company price (yesterday)")
    cost: float = Field(..., description="Today's cost per liter")
    comp1_price: float
    comp2_price: float
    comp3_price: float


@app.on_event("startup")
def startup_load_artifacts() -> None:
    """
    Load historical data, feature table, and the trained model into memory.
    """
    global _history_features, _model, _feature_cols

    raw_df = load_raw_history()
    clean_df = clean_history(raw_df)
    _history_features = build_feature_table(clean_df)

    _model, _feature_cols = load_model_and_config()


@app.get("/health")
def health() -> dict:
    """Simple health check endpoint."""
    return {"status": "ok"}


@app.post("/recommend_price")
def recommend_price_endpoint(req: TodayRequest) -> dict:
    """
    Recommend the optimal price for the given 'today' context.
    """
    if _history_features is None or _model is None or _feature_cols is None:
        raise HTTPException(status_code=500, detail="Model artifacts not loaded.")

    try:
        result = recommend_price_for_today(
            today=req.dict(),
            history_features=_history_features,
            model=_model,
            feature_cols=_feature_cols,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return result
