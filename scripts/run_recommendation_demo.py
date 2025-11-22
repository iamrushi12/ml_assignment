"""
CLI demo script:
- loads today_example.json
- loads history and builds feature table
- loads model & recommends today's optimal price
"""

import json
import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_pipeline import load_raw_history, clean_history  # noqa: E402
from src.features import build_feature_table  # noqa: E402
from src.pricing import recommend_price  # noqa: E402


def main() -> None:
    today_path = Path("data/raw/today_example.json")
    if not today_path.exists():
        raise FileNotFoundError(f"today_example.json not found at {today_path}")

    with today_path.open("r") as f:
        today = json.load(f)

    raw_df = load_raw_history()
    clean_df = clean_history(raw_df)
    history_features = build_feature_table(clean_df)

    result = recommend_price(today, history_features)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
