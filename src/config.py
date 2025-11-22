"""
Global configuration for paths, business rules, and training settings.
"""

# Data paths (relative to project root)
DATA_RAW_HISTORY_PATH = "data/raw/oil_retail_history.csv"
DATA_PROCESSED_DIR = "data/processed"

MODEL_PATH = "models/volume_model.pkl"
FEATURE_CONFIG_PATH = "models/feature_config.json"
TRAINING_METADATA_PATH = "models/training_metadata.json"

# Business rules for pricing
# --------------------------
# Maximum absolute change allowed vs yesterday's price
MAX_ABS_PRICE_CHANGE = 1.0  # e.g. ±1 currency unit

# Minimum profit margin per liter (price - cost)
MIN_MARGIN_PER_LITER = 0.5

# Maximum allowed gap above average competitor price
COMPETITIVE_MAX_DELTA = 0.5

# Price search grid
PRICE_GRID_STEP = 0.05
MIN_PRICE = 50.0
MAX_PRICE = 120.0

# Training configuration
VALIDATION_FRACTION = 0.2  # last 20% of time series as validation
RANDOM_STATE = 42
