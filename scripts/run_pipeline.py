"""
CLI script to run the full pipeline:
- load raw history
- clean and feature-engineer
- train volume model
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import load_raw_history, clean_history  # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.modeling import train_volume_model  # noqa: E402
from src.config import DATA_PROCESSED_DIR  # noqa: E402


def main() -> None:
    os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)

    print("Loading raw history...")
    raw_df = load_raw_history()

    print("Cleaning history...")
    clean_df = clean_history(raw_df)

    print("Building feature table...")
    feat_df = build_feature_table(clean_df)

    feat_path = Path(DATA_PROCESSED_DIR) / "features.parquet"
    feat_df.to_parquet(feat_path, index=False)
    print(f"Saved feature table to {feat_path}")

    print("Training volume model...")
    _, metrics = train_volume_model(feat_df)
    print("Training complete. Validation metrics:", metrics)


if __name__ == "__main__":
    main()
