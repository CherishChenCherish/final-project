# BTC Market Regime Classification — CPSC 381 Final Project

**Author:** Cherish (Xinlong) Chen, Yale SOM
**Course:** CPSC 381/581 — Introduction to Machine Learning, Spring 2026
**Instructor:** Prof. Alex Wong

---

## Research Question

Do on-chain blockchain features (MVRV, hash rate, exchange flows, miner revenue,
active-address counts) provide additional predictive value over price-only
technical indicators for classifying Bitcoin market regimes?

A central methodological concern is **MVRV self-leakage**: the most popular
on-chain regime indicator (MVRV ratio) is also the most natural label source.
Using MVRV both as feature and as label generates spuriously high accuracy.
This project explicitly addresses this issue with two parallel label sets and
an MVRV-ablation feature set.

---

## Repository Layout

```
final/
├── data/
│   ├── raw/                    # Raw downloads (price + on-chain)
│   └── processed/              # Final feature + label table
├── src/
│   ├── data_collection.py      # Pulls Yahoo Finance + CoinMetrics
│   ├── feature_engineering.py  # Builds features and 4 label sets
│   ├── models.py               # Walk-forward training of 5 sklearn models
│   ├── temporal_cnn.py         # 1D CNN training on the same folds
│   └── visualization.py        # Generates all 12 figures
├── models/                     # Result JSONs (full / sensitivity / cnn)
├── figures/                    # All PNG figures used in the report
├── report/                     # Final report (markdown + PDF)
├── requirements.txt
└── README.md
```

---

## Quickstart — Reproduce All Results

### 1. Environment

```bash
cd final
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Tested on Python 3.11, macOS (Apple Silicon, MPS backend for PyTorch).

### 2. Run the full pipeline

The four scripts must run in this order:

```bash
# Step 1 — download raw data (~30 seconds, requires network)
python src/data_collection.py

# Step 2 — feature engineering and label construction (~5 seconds)
python src/feature_engineering.py

# Step 3 — train and evaluate all sklearn models (~3 minutes)
python src/models.py

# Step 4 — train 1D CNN on all folds (~5 minutes on Apple MPS,
#                                     ~15 minutes on CPU)
python src/temporal_cnn.py

# Step 5 — regenerate all figures (~30 seconds)
python src/visualization.py
```

Expected outputs after a full run:

| Path | Content |
|------|---------|
| `data/raw/btc_price.csv` | OHLCV from Yahoo Finance (~4,100 rows) |
| `data/raw/btc_onchain.csv` | CoinMetrics on-chain metrics (~4,100 rows) |
| `data/processed/btc_features_labels.csv` | 55-column merged dataset (4,081 rows) |
| `models/results_full.json` | Sklearn results (90 records: 2 labels × 3 feature sets × 5 models × 3 folds) |
| `models/sensitivity_results.json` | Conservative/aggressive MVRV thresholds (24 records) |
| `models/cnn_results.json` | CNN walk-forward results (12 records) |
| `figures/*.png` | 12 figures used in the report |

### 3. Sanity checks

After step 2 the script will print top-5 extreme forward-return days and label
distributions. After step 3 it will print mean ± std macro-F1 per model per
label.

---

## What Each Component Does

### Data (`data_collection.py`)

- **Price**: BTC-USD daily OHLCV from Yahoo Finance, 2015-01-01 to 2026-03-05
- **On-chain**: CoinMetrics Community API (free tier), metrics include
  `CapMVRVCur`, `AdrActCnt`, `HashRate`, `IssTotUSD`, exchange flows, etc.

### Features and labels (`feature_engineering.py`)

Two label sets are constructed:

1. **MVRV-threshold labels** (`regime_base`, `regime_conservative`, `regime_aggressive`):
   - 0 = Accumulation (MVRV < 1.0 / 0.8 / 1.2)
   - 1 = Expansion (middle)
   - 2 = Distribution (MVRV > 3.5 / 3.8 / 3.0)

2. **Forward-return labels** (`regime_price`) — clean control without MVRV:
   - 0 = Bear (next-30-day return < −5%)
   - 1 = Sideways (−5% to +10%)
   - 2 = Bull (> +10%)

Three feature sets are defined:

- **A**: 8 price-derived technical indicators (SMA-30, RSI-14, MACD trio,
  Volatility-30d, Returns)
- **B**: A + 19 on-chain features (full set, including MVRV)
- **C**: A + on-chain features **with all MVRV-derived columns removed**
  (ablation to quantify MVRV self-leakage)

### Walk-forward CV by halving cycle

| Fold | Train | Test |
|------|-------|------|
| 1 | Cycle 1 (2015-01–2016-07) | Cycle 2 (2016-07–2020-05) |
| 2 | Cycles 1+2 | Cycle 3 (2020-05–2024-04) |
| 3 | Cycles 1+2+3 | Cycle 4 (2024-04–2026-03) |

Standard scaler is fit on the train portion only. NaN handling is forward-fill
(past-only — no leakage). Folds where the test set has only one class are
**explicitly reported** as trivial, not silently dropped.

### Models

| Model | Library | Hyperparameters |
|-------|---------|-----------------|
| Majority Class | `sklearn.dummy.DummyClassifier` | `strategy='most_frequent'` |
| Ridge | `RidgeClassifier` | α=1.0 |
| Linear SVM | `LinearSVC` | C ∈ {0.01, 0.1, 1, 10}, selected by 3-fold inner expanding-window CV |
| Random Forest | `RandomForestClassifier` | n=300, depth=10, min-leaf=5, balanced class weight |
| XGBoost | `XGBClassifier` | n=300, depth=6, lr=0.05, sub=0.8, col=0.8 |
| 1D CNN | PyTorch (MPS) | 60-day window, 3 Conv1d blocks, BatchNorm, AdaptiveAvgPool, Dropout=0.3, Adam lr=1e-3, 40 epochs, balanced class weights |

All sklearn models use `random_state=42`; PyTorch uses `torch.manual_seed(42)`.

---

## Headline Findings (full discussion in `report/REPORT.md`)

1. **MVRV self-leakage is large.** Under the `regime_base` label, XGBoost on
   feature-set B (with MVRV) reaches mean macro-F1 ≈ 0.81. The same model on
   feature-set C (no MVRV) drops to 0.32 — a gap of ~0.5 attributable purely
   to circular construction.

2. **Without MVRV in features, the on-chain advantage is small but real.**
   On the clean `regime_price` label, CNN reaches macro-F1 = 0.41 on the cycle-4
   test set, vs. 0.21 for the Majority baseline (≈ 2× lift). All sklearn models
   sit between 0.17 and 0.34, vs. Majority 0.11–0.18.

3. **CNN beats sklearn models on the most recent fold.** On `regime_price`
   train-c123-test-c4, CNN macro-F1 = 0.41 vs. XGBoost 0.17 and Random Forest
   0.19, suggesting that the 60-day temporal context captures signal that
   point-in-time tabular models miss.

4. **Cycle-1-only training is fundamentally insufficient.** Cycle 1 contains
   only 555 days post-feature-construction and lacks any Distribution-class
   examples under MVRV labels, leading to first-fold collapse for every model.

---

## Requirements

```
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3
xgboost>=2.0
torch>=2.0
matplotlib>=3.7
seaborn>=0.12
shap>=0.44
yfinance>=0.2
requests>=2.31
```

See `requirements.txt` for the exact pinned versions.

---

## Notes on Reproducibility

- All experiments use `random_state=42`.
- The pipeline is fully deterministic except for PyTorch CNN training
  (CUDA/MPS kernels can introduce small non-determinism; results vary by
  ±0.02 macro-F1 across re-runs).
- The CoinMetrics free API is rate-limited; if `data_collection.py` fails on
  some metrics, the cached `data/raw/*.csv` files in this repo can be used
  directly to reproduce all downstream results.
