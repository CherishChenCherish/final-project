"""
Step 2: Feature Engineering & Label Construction
- Compute technical indicators from price data
- Compute Puell Multiple from miner revenue
- Construct regime labels using MVRV thresholds
- Three threshold variants for sensitivity analysis
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')


# ============================================================
# 1. Technical Indicators (price-derived features)
# ============================================================
def compute_technical_indicators(price_df):
    """Compute price-derived features: SMA, RSI, MACD, volatility, returns."""
    df = price_df.copy()

    close = df['Close'].squeeze() if hasattr(df['Close'], 'squeeze') else df['Close']

    # 30-day simple moving average
    df['SMA_30'] = close.rolling(30).mean()

    # Price relative to SMA
    df['Price_SMA_Ratio'] = close / df['SMA_30']

    # RSI-14
    delta = close.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    ema12 = close.ewm(span=12).mean()
    ema26 = close.ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # 30-day rolling volatility (annualized)
    df['Returns'] = close.pct_change()
    df['Volatility_30d'] = df['Returns'].rolling(30).std() * np.sqrt(365)

    # 30-day forward return (for regression sub-task)
    df['Forward_Return_30d'] = close.shift(-30) / close - 1

    # BTC dominance placeholder (will need external data or approximation)
    # df['BTC_Dominance'] = ...

    return df


# ============================================================
# 2. On-chain Feature Processing
# ============================================================
def process_onchain_features(onchain_df):
    """Process raw on-chain data into model features."""
    df = onchain_df.copy()

    # Puell Multiple (already computed in data_collection if IssTotUSD available)
    # If not present, skip
    if 'Puell_Multiple' not in df.columns and 'IssTotUSD' in df.columns:
        df['Puell_Multiple'] = df['IssTotUSD'] / df['IssTotUSD'].rolling(365).mean()

    # MVRV Z-Score proxy using MVRV ratio itself (no Realized Cap available)
    if 'CapMVRVCur' in df.columns:
        mvrv = df['CapMVRVCur']
        df['MVRV_Z'] = (mvrv - mvrv.rolling(365).mean()) / mvrv.rolling(365).std()

    # Log transforms for skewed metrics
    for col in ['AdrActCnt', 'HashRate', 'CapMrktCurUSD', 'TxCnt', 'TxTfrCnt', 'IssTotUSD']:
        if col in df.columns:
            df[f'{col}_log'] = np.log1p(df[col])

    # Rate of change (7d, 30d) for key metrics
    for col in ['CapMVRVCur', 'AdrActCnt', 'HashRate', 'TxCnt']:
        if col in df.columns:
            df[f'{col}_roc_7d'] = df[col].pct_change(7)
            df[f'{col}_roc_30d'] = df[col].pct_change(30)

    # Exchange flow features (already computed if available)
    if 'FlowInExUSD' in df.columns and 'FlowOutExUSD' in df.columns:
        if 'NetFlowExUSD' not in df.columns:
            df['NetFlowExUSD'] = df['FlowInExUSD'] - df['FlowOutExUSD']
        # 7-day rolling net flow
        df['NetFlowExUSD_7d'] = df['NetFlowExUSD'].rolling(7).mean()

    # Exchange supply ratio
    if 'SplyExNtv' in df.columns and 'SplyCur' in df.columns:
        if 'ExchangeSupplyRatio' not in df.columns:
            df['ExchangeSupplyRatio'] = df['SplyExNtv'] / df['SplyCur']

    return df


# ============================================================
# 3. Label Construction using MVRV thresholds
# ============================================================

# Three threshold variants for sensitivity analysis
THRESHOLD_VARIANTS = {
    'base':         {'accum_upper': 1.0, 'distrib_lower': 3.5},
    'conservative': {'accum_upper': 0.8, 'distrib_lower': 3.8},
    'aggressive':   {'accum_upper': 1.2, 'distrib_lower': 3.0},
}


def construct_labels(df, mvrv_col='CapMVRVCur', variant='base'):
    """
    Assign regime labels based on MVRV thresholds.
    0 = Accumulation (MVRV < threshold)
    1 = Expansion (middle)
    2 = Distribution (MVRV > threshold)
    """
    thresholds = THRESHOLD_VARIANTS[variant]
    mvrv = df[mvrv_col]

    labels = pd.Series(1, index=df.index, name='regime')  # default: Expansion
    labels[mvrv < thresholds['accum_upper']] = 0   # Accumulation
    labels[mvrv > thresholds['distrib_lower']] = 2  # Distribution

    return labels


def construct_all_label_variants(df, mvrv_col='CapMVRVCur'):
    """Construct labels for all three threshold variants."""
    for variant in THRESHOLD_VARIANTS:
        df[f'regime_{variant}'] = construct_labels(df, mvrv_col, variant)
    return df


# ============================================================
# 4. Define halving cycle boundaries (for walk-forward splits)
# ============================================================
HALVING_CYCLES = {
    'cycle1': ('2013-01-01', '2016-07-08'),   # Before 2nd halving
    'cycle2': ('2016-07-09', '2020-05-10'),   # 2nd to 3rd halving
    'cycle3': ('2020-05-11', '2024-04-19'),   # 3rd to 4th halving
    'cycle4': ('2024-04-20', '2026-12-31'),   # 4th halving onwards
}


def assign_cycle(date):
    """Assign a halving cycle number to a date."""
    for i, (name, (start, end)) in enumerate(HALVING_CYCLES.items()):
        if pd.Timestamp(start) <= date <= pd.Timestamp(end):
            return i + 1
    return None


# ============================================================
# 5. Merge everything into final dataset
# ============================================================
def build_final_dataset():
    """Load raw data, compute features, construct labels, save processed dataset."""
    print("Building final dataset...")

    # Load raw data
    # yfinance saves multi-level headers: skip ticker row, use first row as columns
    price_df = pd.read_csv(os.path.join(RAW_DIR, 'btc_price.csv'), header=[0, 1], index_col=0, parse_dates=True)
    # Flatten multi-level columns to just the price type (Close, High, etc.)
    price_df.columns = price_df.columns.get_level_values(0)
    price_df.index.name = 'Date'
    onchain_df = pd.read_csv(os.path.join(RAW_DIR, 'btc_onchain.csv'), index_col=0, parse_dates=True)

    # Compute features
    price_feat = compute_technical_indicators(price_df)
    onchain_feat = process_onchain_features(onchain_df)

    # Merge on date index
    merged = price_feat.join(onchain_feat, how='inner')

    # Construct labels (three variants)
    merged = construct_all_label_variants(merged)

    # Assign halving cycle
    merged['cycle'] = merged.index.map(assign_cycle)

    # Z-score normalize numeric features (fit on training data later, just prep here)
    # Save
    merged.to_csv(os.path.join(PROCESSED_DIR, 'btc_features_labels.csv'))
    print(f"  Saved {len(merged)} rows to data/processed/btc_features_labels.csv")

    # Print label distribution
    for variant in THRESHOLD_VARIANTS:
        col = f'regime_{variant}'
        counts = merged[col].value_counts().sort_index()
        print(f"\n  Label distribution ({variant}):")
        print(f"    0 (Accumulation): {counts.get(0, 0)}")
        print(f"    1 (Expansion):    {counts.get(1, 0)}")
        print(f"    2 (Distribution): {counts.get(2, 0)}")

    return merged


if __name__ == '__main__':
    build_final_dataset()
