# Fuel Price Optimizer

End-to-end ML system to recommend daily fuel prices that maximize profit,
based on historical retail data and competitor prices.

This repo contains:
- Data pipeline (ingestion, cleaning, feature engineering)
- Volume prediction model (XGBoost)
- Pricing engine with business guardrails
- FastAPI service to serve daily price recommendations
