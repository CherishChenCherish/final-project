"""
Step 2: Feature Engineering & Label Construction
- Compute technical indicators from price data
- Process on-chain features (Puell, MVRV-Z, ROC, log-transforms)
- Construct TWO label sets:
    * regime_base / regime_conservative / regime_aggressive  (MVRV thresholds)
    * regime_price                                            (forward return thresholds)
- The two label sets serve different purposes:
    * MVRV labels test "can we recover MVRV-defined regimes from features?"
        (suffers from circularity if MVRV is also a feature)
    * Price labels test "can features predict near-future return direction?"
        (no circularity, clean test of on-chain signal)
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

# ----------------------------------------------------------------
# Pre-registered thresholds (chosen BEFORE seeing model results)
# ----------------------------------------------------------------
# MVRV thresholds (three variants for sensitivity analysis)
MVRV_THRESHOLDS = {
    'base':         {'accum_upper': 1.0, 'distrib_lower': 3.5},
    'conservative': {'accum_upper': 0.8, 'distrib_lower': 3.8},
    'aggressive':   {'accum_upper': 1.2, 'distrib_lower': 3.0},
}

# Forward 30d return thresholds (fixed, interpretable)
PRICE_LABEL_THRESHOLDS = {
    'bear_upper':  -0.05,   # < -5% in next 30d => Bear (0)
    'bull_lower':   0.10,   # > +10% in next 30d => Bull (2)
    # Sideways (1) = in between
}

# Halving cycles (used for walk-forward split)
HALVING_CYCLES = {
    'cycle1': ('2013-01-01', '2016-07-08'),
    'cycle2': ('2016-07-09', '2020-05-10'),
    'cycle3': ('2020-05-11', '2024-04-19'),
    'cycle4': ('2024-04-20', '2026-12-31'),
}


# ============================================================
# 1. Technical Indicators (price-derived features)
# ============================================================
def compute_technical_indicators(price_df):
    """Compute price-derived features using ONLY past information."""
    df = price_df.copy()
    close = df['Close'].squeeze() if hasattr(df['Close'], 'squeeze') else df['Close']

    df['SMA_30'] = close.rolling(30).mean()
    df['Price_SMA_Ratio'] = close / df['SMA_30']

    # RSI-14
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9) — explicit min_periods to avoid using almost-empty EMA at the start
    ema12 = close.ewm(span=12, min_periods=12).mean()
    ema26 = close.ewm(span=26, min_periods=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, min_periods=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Returns + volatility (annualized)
    df['Returns'] = close.pct_change()
    df['Volatility_30d'] = df['Returns'].rolling(30).std() * np.sqrt(365)

    # Forward 30-day return — used ONLY as label/regression target, NEVER as feature
    df['Forward_Return_30d'] = close.shift(-30) / close - 1

    return df


# ============================================================
# 2. On-chain Feature Processing
# ============================================================
def process_onchain_features(onchain_df):
    """Process raw on-chain data into model features."""
    df = onchain_df.copy()

    # Puell Multiple = miner revenue / 365d MA
    if 'Puell_Multiple' not in df.columns and 'IssTotUSD' in df.columns:
        df['Puell_Multiple'] = df['IssTotUSD'] / df['IssTotUSD'].rolling(365).mean()

    # MVRV Z-Score (proxy: rolling z of MVRV ratio, since Realized Cap z requires longer history)
    if 'CapMVRVCur' in df.columns:
        mvrv = df['CapMVRVCur']
        df['MVRV_Z'] = (mvrv - mvrv.rolling(365).mean()) / mvrv.rolling(365).std()

    # Log transforms for skewed metrics
    for col in ['AdrActCnt', 'HashRate', 'CapMrktCurUSD', 'TxCnt', 'TxTfrCnt', 'IssTotUSD']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])

    # Rate of change features (7d, 30d)
    for col in ['CapMVRVCur', 'AdrActCnt', 'HashRate', 'TxCnt']:
        if col in df.columns:
            df[f'{col}_roc_7d'] = df[col].pct_change(7)
            df[f'{col}_roc_30d'] = df[col].pct_change(30)

    # Exchange flows
    if 'FlowInExUSD' in df.columns and 'FlowOutExUSD' in df.columns:
        if 'NetFlowExUSD' not in df.columns:
            df['NetFlowExUSD'] = df['FlowInExUSD'] - df['FlowOutExUSD']
        df['NetFlowExUSD_7d'] = df['NetFlowExUSD'].rolling(7).mean()

    if 'SplyExNtv' in df.columns and 'SplyCur' in df.columns:
        if 'ExchangeSupplyRatio' not in df.columns:
            df['ExchangeSupplyRatio'] = df['SplyExNtv'] / df['SplyCur']

    return df


# ============================================================
# 3. Label Construction
# ============================================================
def construct_mvrv_labels(df, mvrv_col='CapMVRVCur', variant='base'):
    """
    MVRV-threshold regime labels.
    0 = Accumulation, 1 = Expansion, 2 = Distribution
    NOTE: If MVRV-derived features are used to predict these labels,
          performance will be inflated by circularity.
    """
    th = MVRV_THRESHOLDS[variant]
    mvrv = df[mvrv_col]
    labels = pd.Series(1, index=df.index, dtype='Int64', name=f'regime_{variant}')
    labels[mvrv < th['accum_upper']] = 0
    labels[mvrv > th['distrib_lower']] = 2
    # MVRV NaN => label NaN (so train/test don't include garbage)
    labels[mvrv.isna()] = pd.NA
    return labels


def construct_price_labels(df, fwd_col='Forward_Return_30d'):
    """
    Forward-return regime labels (clean — no MVRV circularity).
    0 = Bear (< -5% next 30d)
    1 = Sideways (-5% to +10%)
    2 = Bull   (> +10%)
    """
    th = PRICE_LABEL_THRESHOLDS
    fwd = df[fwd_col]
    labels = pd.Series(1, index=df.index, dtype='Int64', name='regime_price')
    labels[fwd < th['bear_upper']] = 0
    labels[fwd > th['bull_lower']] = 2
    labels[fwd.isna()] = pd.NA
    return labels


# ============================================================
# 4. Cycle assignment
# ============================================================
def assign_cycle(date):
    """Assign halving cycle (1-4) to a date."""
    for i, (name, (start, end)) in enumerate(HALVING_CYCLES.items()):
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return i + 1
    return None


# ============================================================
# 5. Build final dataset (with assertions before save — Iron Law #1)
# ============================================================
def build_final_dataset():
    print("Building final dataset...")

    # Load raw
    price_df = pd.read_csv(
        os.path.join(RAW_DIR, 'btc_price.csv'),
        header=[0, 1], index_col=0, parse_dates=True
    )
    price_df.columns = price_df.columns.get_level_values(0)
    price_df.index.name = 'Date'
    onchain_df = pd.read_csv(
        os.path.join(RAW_DIR, 'btc_onchain.csv'),
        index_col=0, parse_dates=True
    )

    # Compute features
    price_feat = compute_technical_indicators(price_df)
    onchain_feat = process_onchain_features(onchain_df)

    # Merge on date
    merged = price_feat.join(onchain_feat, how='inner')

    # Construct ALL label sets
    for variant in MVRV_THRESHOLDS:
        merged[f'regime_{variant}'] = construct_mvrv_labels(merged, variant=variant)
    merged['regime_price'] = construct_price_labels(merged)

    # Assign cycle
    merged['cycle'] = merged.index.map(assign_cycle)

    # ----- ASSERT block (Iron Law #1) -----
    assert merged.index.is_monotonic_increasing, "Date index must be sorted ascending"
    assert merged.index.is_unique, "Duplicate dates found"
    assert 'CapMVRVCur' in merged.columns, "Missing MVRV column"
    assert 'Forward_Return_30d' in merged.columns, "Missing forward-return column"

    # Each label set must have >=2 unique values overall
    for col in ['regime_base', 'regime_conservative', 'regime_aggressive', 'regime_price']:
        n_unique = merged[col].dropna().nunique()
        assert n_unique >= 2, f"{col} has only {n_unique} unique label(s)"

    # Forward_Return_30d must have NaN for the LAST 30 days (no future info available)
    last_30_nans = merged['Forward_Return_30d'].iloc[-30:].isna().sum()
    assert last_30_nans == 30, (
        f"Forward_Return_30d should be NaN for last 30 days "
        f"(got {last_30_nans} NaN, expected 30)"
    )

    # Cycle assignment sanity
    assert merged['cycle'].isin([1, 2, 3, 4]).all(), "Some rows have invalid cycle"

    # Print top-5 extremes (sanity check)
    print("\n  Top-5 highest forward-30d return:")
    print(merged.nlargest(5, 'Forward_Return_30d')[['Close', 'Forward_Return_30d', 'CapMVRVCur']])
    print("\n  Top-5 lowest forward-30d return:")
    print(merged.nsmallest(5, 'Forward_Return_30d')[['Close', 'Forward_Return_30d', 'CapMVRVCur']])

    # Save
    merged.to_csv(os.path.join(PROCESSED_DIR, 'btc_features_labels.csv'))
    print(f"\n  Saved {len(merged)} rows × {len(merged.columns)} cols to "
          f"data/processed/btc_features_labels.csv")

    # Print label distributions
    print("\n  === Label distributions ===")
    for col in ['regime_base', 'regime_conservative', 'regime_aggressive', 'regime_price']:
        counts = merged[col].value_counts(dropna=False).sort_index()
        total = counts.sum()
        print(f"\n  {col}:")
        for k, v in counts.items():
            label_name = {0: 'Bear/Accum', 1: 'Sideways/Exp', 2: 'Bull/Distrib', pd.NA: 'NaN'}.get(k, str(k))
            print(f"    {k} ({label_name}): {v} ({v/total*100:.1f}%)")

    print("\n  === Label x Cycle (regime_price) ===")
    print(merged.groupby('cycle')['regime_price'].value_counts().unstack(fill_value=0))

    return merged


if __name__ == '__main__':
    build_final_dataset()
