# Fuel Price predication 

End-to-end ML system to recommend daily fuel prices that maximize profit,
based on historical retail data and competitor prices.

## Problem overview

A retail fuel company sets one price per day. Profit for a day is:

> profit = (price - cost) × volume

Volume depends on the company price, competitor prices, and seasonality.
The goal is to:

- Predict expected **volume** given a candidate price and context
- Search over candidate prices to find the one that maximizes predicted profit
- Enforce business guardrails (max price move per day, minimum margin, etc.)

## Repository structure

```text
fuel-price-optimizer/
  data/
    raw/
      oil_retail_history.csv   # 2 years of daily history
      today_example.json       # example of today’s known inputs
    processed/                 # generated feature tables
  models/                      # trained model & metadata
  notebooks/                   # EDA notebook(s)
  src/
    api/                       # FastAPI service
    config.py                  # paths & configuration
    data_pipeline.py           # ingestion & cleaning
    features.py                # feature engineering
    modeling.py                # training & evaluation
    pricing.py                 # pricing engine (profit optimization)
  scripts/
    run_pipeline.py            # run ETL + training
    run_recommendation_demo.py # demo: recommend price for today_example.json
  tests/                       # basic automated tests


#Outputs:-

<img width="924" height="447" alt="image" src="https://github.com/user-attachments/assets/402fd990-1abf-4217-aa91-68a72c0dc8f3" />

<img width="1068" height="543" alt="image" src="https://github.com/user-attachments/assets/7137197f-b7c0-4416-b527-ce9050ddd242" />

