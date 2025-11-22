"""
Model training and evaluation for volume prediction.
"""

import json
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

from .config import (
    MODEL_PATH,
    FEATURE_CONFIG_PATH,
    TRAINING_METADATA_PATH,
    VALIDATION_FRACTION,
    RANDOM_STATE,
)
from .features import FEATURE_COLUMNS, TARGET_COLUMN


def time_based_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Time-ordered train/validation split:
    - sort by date
    - use last VALIDATION_FRACTION fraction as validation
    """
    df = df.sort_values("date").reset_index(drop=True)
    n_rows = len(df)
    if n_rows < 10:
        raise ValueError("Not enough data for a meaningful train/validation split.")

    split_idx = int(n_rows * (1.0 - VALIDATION_FRACTION))
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    return train_df, val_df


def train_volume_model(feature_df: pd.DataFrame):
    """
    Train an XGBoost regression model to predict daily volume.

    Saves:
      - trained model (MODEL_PATH)
      - feature configuration (FEATURE_CONFIG_PATH)
      - training metadata & metrics (TRAINING_METADATA_PATH)
    """
    train_df, val_df = time_based_split(feature_df)

    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[TARGET_COLUMN]
    X_val = val_df[FEATURE_COLUMNS]
    y_val = val_df[TARGET_COLUMN]

    model = XGBRegressor(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)

    # Older sklearn versions don't support squared=False, so compute RMSE manually
    mse = mean_squared_error(y_val, y_pred)
    rmse = float(np.sqrt(mse))

    metrics = {"mae": float(mae), "rmse": rmse}

    # Persist model
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    # Persist feature configuration
    with open(FEATURE_CONFIG_PATH, "w") as f:
        json.dump(
            {"feature_columns": FEATURE_COLUMNS, "target_column": TARGET_COLUMN},
            f,
        )

    # Persist training metadata
    metadata = {
        "metrics": metrics,
        "n_train": int(len(X_train)),
        "n_val": int(len(X_val)),
    }
    with open(TRAINING_METADATA_PATH, "w") as f:
        json.dump(metadata, f)

    return model, metrics
