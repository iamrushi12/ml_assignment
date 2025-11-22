"""
Basic smoke tests to ensure the core modules at least import and
very simple functionality behaves as expected.
"""

from src.config import RANDOM_STATE
from src.data_pipeline import clean_history
import pandas as pd


def test_random_state_is_int():
    assert isinstance(RANDOM_STATE, int)


def test_clean_history_filters_invalid_rows():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2024-01-01", "2024-01-02"]),
            "price": [100.0, -5.0],
            "cost": [90.0, 80.0],
            "comp1_price": [99.0, 99.0],
            "comp2_price": [99.0, 99.0],
            "comp3_price": [99.0, 99.0],
            "volume": [1000, 2000],
        }
    )

    clean_df = clean_history(df)
    assert len(clean_df) == 1
    assert float(clean_df.iloc[0]["price"]) == 100.0
