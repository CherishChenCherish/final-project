"""
Step 1: Data Collection
- CoinMetrics Community API: on-chain metrics
- Yahoo Finance (yfinance): price data + technical indicators
Date range: 2015-01-01 to 2026-03-05
"""

import pandas as pd
import numpy as np
import yfinance as yf
import requests
from datetime import datetime
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

START_DATE = '2015-01-01'
END_DATE = '2026-03-05'


# ============================================================
# 1. Yahoo Finance: BTC price data
# ============================================================
def fetch_price_data():
    """Download BTC-USD OHLCV data from Yahoo Finance."""
    print("Fetching BTC price data from Yahoo Finance...")
    btc = yf.download('BTC-USD', start=START_DATE, end=END_DATE)
    btc.index = pd.to_datetime(btc.index).tz_localize(None)
    btc.to_csv(os.path.join(RAW_DIR, 'btc_price.csv'))
    print(f"  Saved {len(btc)} rows to data/raw/btc_price.csv")
    return btc


# ============================================================
# 2. CoinMetrics Community API: on-chain metrics
# ============================================================
COINMETRICS_BASE = "https://community-api.coinmetrics.io/v4"

# Key on-chain metrics from the proposal
ONCHAIN_METRICS = [
    'CapMrktCurUSD',      # Market Cap
    'CapRealUSD',         # Realized Cap
    'CapMVRVCur',         # MVRV ratio
    'NVTAdj',             # NVT (Network Value to Transactions)
    'SplyCntCDD90d',      # Coin Days Destroyed (90d)
    'AdrActCnt',          # Active Addresses
    'HashRate',           # Hash Rate
    'RevUSD',             # Miner Revenue (for Puell Multiple calc)
    'SplyActEver',        # Supply ever active
]


def fetch_coinmetrics_metric(metric, asset='btc'):
    """Fetch a single metric from CoinMetrics Community API."""
    url = f"{COINMETRICS_BASE}/timeseries/asset-metrics"
    params = {
        'assets': asset,
        'metrics': metric,
        'start_time': START_DATE,
        'end_time': END_DATE,
        'page_size': 10000,
        'frequency': '1d',
    }
    all_data = []
    while True:
        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"  Warning: failed to fetch {metric}, status={resp.status_code}")
            return pd.DataFrame()
        data = resp.json()
        rows = data.get('data', [])
        all_data.extend(rows)
        next_page = data.get('next_page_url')
        if not next_page:
            break
        url = next_page
        params = {}
    if not all_data:
        return pd.DataFrame()
    df = pd.DataFrame(all_data)
    df['time'] = pd.to_datetime(df['time']).dt.tz_localize(None)
    df = df.set_index('time')
    if metric in df.columns:
        df[metric] = pd.to_numeric(df[metric], errors='coerce')
    return df[[metric]] if metric in df.columns else df


def fetch_all_onchain():
    """Fetch all on-chain metrics and merge into one DataFrame."""
    print("Fetching on-chain metrics from CoinMetrics...")
    frames = []
    for metric in ONCHAIN_METRICS:
        print(f"  Fetching {metric}...")
        df = fetch_coinmetrics_metric(metric)
        if not df.empty:
            frames.append(df)
    if not frames:
        print("  Warning: no on-chain data fetched!")
        return pd.DataFrame()
    onchain = pd.concat(frames, axis=1)
    onchain.to_csv(os.path.join(RAW_DIR, 'btc_onchain.csv'))
    print(f"  Saved {len(onchain)} rows to data/raw/btc_onchain.csv")
    return onchain


# ============================================================
# 3. SOPR (Spent Output Profit Ratio) - separate source
# ============================================================
def fetch_sopr():
    """
    SOPR from CoinMetrics or Glassnode free tier.
    Falls back to NaN if unavailable.
    """
    print("Fetching SOPR...")
    df = fetch_coinmetrics_metric('SplySOPR')
    if df.empty:
        print("  SOPR not available from CoinMetrics community API, will compute proxy later.")
    else:
        df.to_csv(os.path.join(RAW_DIR, 'btc_sopr.csv'))
    return df


# ============================================================
# Main
# ============================================================
def main():
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    price_df = fetch_price_data()
    onchain_df = fetch_all_onchain()
    sopr_df = fetch_sopr()

    print("\n=== Data Collection Summary ===")
    print(f"Price data: {len(price_df)} days, columns: {list(price_df.columns)}")
    print(f"On-chain data: {len(onchain_df)} days, columns: {list(onchain_df.columns)}")
    if not sopr_df.empty:
        print(f"SOPR data: {len(sopr_df)} days")
    print("Done! Raw data saved to data/raw/")


if __name__ == '__main__':
    main()
