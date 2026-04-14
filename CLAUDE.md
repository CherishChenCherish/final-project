# BTC Market Regime Classification (CPSC 381 Final)

## Quick Commands
- Activate env: `source venv/bin/activate`
- Run all models: `python src/models.py`
- Build features: `python src/feature_engineering.py`
- Collect data: `python src/data_collection.py`

## Project Context
- Yale CPSC 381 Machine Learning final project
- Student: Cherish Chen, Yale SOM
- Research question: Do on-chain features improve BTC regime classification?
- Uses walk-forward expanding window (NOT k-fold) for time series
- Key concern: MVRV circularity — MVRV used in both labels and features

## Code Style
- Python 3.11
- pandas, numpy, sklearn, xgboost, pytorch
- Chinese comments OK
- Type hints preferred but not required

## Data
- Price: Yahoo Finance (yfinance), 2015-01 to 2026-03
- On-chain: CoinMetrics Community API (free tier, some metrics unavailable)
- Processed dataset: data/processed/btc_features_labels.csv

## Important Notes
- Distribution class is severely imbalanced (1.2% under base MVRV threshold)
- Cycle 4 (2024-2026) has only Expansion under MVRV labels
- Price-momentum labels (regime_price) avoid circularity and have 3 valid folds
- Always use `matplotlib.use('Agg')` in scripts (no display server)
