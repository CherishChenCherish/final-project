"""
Step 3 & 4: Model Training & Evaluation Pipeline
================================================

Walk-forward expanding-window cross-validation, evaluated on TWO label sets:

  1. regime_base / regime_conservative / regime_aggressive  (MVRV thresholds)
  2. regime_price                                            (forward-return thresholds)

Models compared:
  - Majority Class baseline (DummyClassifier)
  - Ridge (RidgeClassifier)
  - Linear SVM (LinearSVC, with C grid-searched via expanding-window CV on train)
  - Random Forest
  - XGBoost

Feature sets:
  - A: price-only
  - B: price + on-chain (full)
  - C: price + on-chain WITHOUT MVRV (for circularity ablation)

For each (label, feature_set, model, fold) we record:
  - macro_f1, weighted_f1, accuracy
  - per-class precision / recall / f1
  - confusion matrix
  - features actually used (after NaN-drop)
  - n_train, n_test
  - skip_reason if fold was trivial / data-insufficient
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    f1_score, accuracy_score,
    classification_report, confusion_matrix,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RANDOM_STATE = 42

# ============================================================
# Feature groups
# ============================================================
PRICE_FEATURES = [
    'SMA_30', 'Price_SMA_Ratio', 'RSI_14',
    'MACD', 'MACD_Signal', 'MACD_Hist',
    'Volatility_30d', 'Returns',
]

ONCHAIN_FEATURES = [
    'CapMVRVCur', 'MVRV_Z',
    'AdrActCnt', 'HashRate', 'TxCnt',
    'Puell_Multiple', 'NVT_proxy',
    'NetFlowExUSD', 'NetFlowExUSD_7d', 'ExchangeSupplyRatio',
    'CapMVRVCur_roc_7d', 'CapMVRVCur_roc_30d',
    'AdrActCnt_roc_7d', 'AdrActCnt_roc_30d',
    'HashRate_roc_7d', 'HashRate_roc_30d',
    'AdrActCnt_log', 'HashRate_log', 'TxCnt_log',
]

# Ablation: drop ALL MVRV-derived features (including ROC of MVRV)
ONCHAIN_NO_MVRV = [f for f in ONCHAIN_FEATURES if 'MVRV' not in f]

FEATURE_SETS = {
    'A_price_only':       PRICE_FEATURES,
    'B_price_onchain':    PRICE_FEATURES + ONCHAIN_FEATURES,
    'C_no_mvrv_ablation': PRICE_FEATURES + ONCHAIN_NO_MVRV,
}

LABEL_COLS = ['regime_base', 'regime_price']  # main experiments
SENSITIVITY_LABELS = ['regime_conservative', 'regime_aggressive']  # only base model


# ============================================================
# Walk-Forward Splits (expanding window, by halving cycle)
# ============================================================
def get_walk_forward_splits(df, cycle_col='cycle'):
    """Yield (train_idx, test_idx, fold_name) for expanding-window CV by cycle."""
    cycles = sorted(df[cycle_col].dropna().unique())
    for i in range(1, len(cycles)):
        train_cycles = cycles[:i]
        test_cycle = cycles[i]
        train_mask = df[cycle_col].isin(train_cycles)
        test_mask = df[cycle_col] == test_cycle
        yield (
            df.index[train_mask],
            df.index[test_mask],
            f"train_c{''.join(map(str, map(int, train_cycles)))}_test_c{int(test_cycle)}"
        )


# ============================================================
# Linear SVM with C tuned by inner expanding-window CV
# ============================================================
def fit_linear_svm_with_cv(X_train_scaled, y_train):
    """Fit LinearSVC with C selected via 3-fold expanding-window CV on training set."""
    C_grid = [0.01, 0.1, 1.0, 10.0]
    tscv = TimeSeriesSplit(n_splits=3)
    best_C, best_score = 1.0, -np.inf
    for C in C_grid:
        scores = []
        for tr_idx, val_idx in tscv.split(X_train_scaled):
            if y_train.iloc[tr_idx].nunique() < 2 or y_train.iloc[val_idx].nunique() < 2:
                continue
            mdl = LinearSVC(C=C, max_iter=5000, dual='auto', random_state=RANDOM_STATE)
            mdl.fit(X_train_scaled[tr_idx], y_train.iloc[tr_idx])
            preds = mdl.predict(X_train_scaled[val_idx])
            scores.append(f1_score(y_train.iloc[val_idx], preds, average='macro'))
        if scores and np.mean(scores) > best_score:
            best_score = np.mean(scores)
            best_C = C
    final = LinearSVC(C=best_C, max_iter=5000, dual='auto', random_state=RANDOM_STATE)
    final.fit(X_train_scaled, y_train)
    return final, best_C


# ============================================================
# Model factory
# ============================================================
def get_models():
    """Return ordered dict of {model_name: estimator (untrained)}."""
    return {
        'Majority':    DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE),
        'Ridge':       RidgeClassifier(alpha=1.0, random_state=RANDOM_STATE),
        'LinearSVM':   None,  # special-cased (CV-tuned)
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=RANDOM_STATE, n_jobs=-1,
        ),
        'XGBoost':     XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=RANDOM_STATE, n_jobs=-1,
        ),
    }


# ============================================================
# Single fold: train + evaluate one (model, feature_set, label, fold)
# ============================================================
def evaluate_one_fold(df, train_idx, test_idx, fold_name,
                      feature_cols, label_col, model_name):
    """Returns one result dict (always — never silently skips)."""
    available = [c for c in feature_cols if c in df.columns]
    X_train = df.loc[train_idx, available].copy()
    y_train = df.loc[train_idx, label_col].copy()
    X_test = df.loc[test_idx, available].copy()
    y_test = df.loc[test_idx, label_col].copy()

    # Drop columns >50% NaN in train
    nan_frac = X_train.isnull().mean()
    drop_cols = nan_frac[nan_frac > 0.5].index.tolist()
    X_train = X_train.drop(columns=drop_cols)
    X_test = X_test.drop(columns=drop_cols)

    # Forward-fill (PAST-only — no leakage), then drop residual NaN rows
    X_train = X_train.ffill()
    X_test = X_test.ffill()
    train_valid = X_train.dropna().index.intersection(y_train.dropna().index)
    test_valid = X_test.dropna().index.intersection(y_test.dropna().index)
    X_train, y_train = X_train.loc[train_valid], y_train.loc[train_valid].astype(int)
    X_test, y_test = X_test.loc[test_valid], y_test.loc[test_valid].astype(int)

    base_record = {
        'fold': fold_name,
        'label': label_col,
        'model': model_name,
        'n_train': int(len(X_train)),
        'n_test': int(len(X_test)),
        'features_used': list(X_train.columns),
        'train_class_dist': y_train.value_counts().sort_index().to_dict() if len(y_train) else {},
        'test_class_dist': y_test.value_counts().sort_index().to_dict() if len(y_test) else {},
    }

    # Skip reasons (recorded, not silently dropped)
    if len(X_train) < 50 or len(X_test) < 20:
        return {**base_record, 'skipped': True,
                'skip_reason': f'insufficient data (train={len(X_train)}, test={len(X_test)})'}
    if y_train.nunique() < 2:
        return {**base_record, 'skipped': True,
                'skip_reason': f'only 1 class in training ({y_train.unique().tolist()})'}
    if y_test.nunique() < 2:
        # Trivial fold: still report majority-class accuracy as the only meaningful number
        majority = int(y_test.mode().iloc[0])
        return {**base_record, 'skipped': True,
                'skip_reason': f'only 1 class in test (class {majority}); test set is trivial'}

    # Standardize (fit on train only — no leakage)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Fit
    extra_info = {}
    if model_name == 'LinearSVM':
        model, best_C = fit_linear_svm_with_cv(X_train_scaled, y_train)
        extra_info['best_C'] = best_C
    elif model_name == 'Majority':
        model = DummyClassifier(strategy='most_frequent', random_state=RANDOM_STATE)
        model.fit(X_train_scaled, y_train)
    else:
        model = get_models()[model_name]
        model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)

    # Metrics
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    weighted_f1 = f1_score(y_test, y_pred, average='weighted')
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1, 2]).tolist()
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    per_class = {}
    for cls_label in ['0', '1', '2']:
        if cls_label in report:
            per_class[cls_label] = {
                'precision': report[cls_label]['precision'],
                'recall':    report[cls_label]['recall'],
                'f1':        report[cls_label]['f1-score'],
                'support':   report[cls_label]['support'],
            }

    return {
        **base_record,
        'skipped': False,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'accuracy': acc,
        'confusion_matrix': cm,
        'per_class': per_class,
        **extra_info,
    }


# ============================================================
# Full grid: all (label, feature_set, model, fold) combinations
# ============================================================
def run_full_experiment(df):
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_results = []
    splits = list(get_walk_forward_splits(df))

    for label_col in LABEL_COLS:
        for fs_name, features in FEATURE_SETS.items():
            print(f"\n{'='*70}")
            print(f"  Label: {label_col}    Feature set: {fs_name}")
            print(f"{'='*70}")
            for model_name in get_models():
                for train_idx, test_idx, fold_name in splits:
                    r = evaluate_one_fold(
                        df, train_idx, test_idx, fold_name,
                        features, label_col, model_name,
                    )
                    all_results.append(r)
                    if r.get('skipped'):
                        print(f"  [SKIP] {fold_name} | {model_name:12s} | {r['skip_reason']}")
                    else:
                        c_msg = f" (C={r['best_C']})" if 'best_C' in r else ''
                        print(f"  {fold_name} | {model_name:12s}{c_msg} | "
                              f"macroF1={r['macro_f1']:.3f} acc={r['accuracy']:.3f} "
                              f"train={r['n_train']} test={r['n_test']}")

    # Save full record
    with open(os.path.join(MODELS_DIR, 'results_full.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  Saved full results to models/results_full.json ({len(all_results)} entries)")

    return all_results


# ============================================================
# Sensitivity analysis: best model × MVRV threshold variants
# ============================================================
def run_sensitivity_analysis(df):
    """Run RandomForest + XGBoost on conservative/aggressive MVRV thresholds."""
    print(f"\n{'='*70}")
    print("  Sensitivity Analysis: alternative MVRV thresholds")
    print(f"{'='*70}")
    results = []
    splits = list(get_walk_forward_splits(df))
    for label_col in SENSITIVITY_LABELS:
        for fs_name, features in [('B_price_onchain', PRICE_FEATURES + ONCHAIN_FEATURES),
                                  ('C_no_mvrv_ablation', PRICE_FEATURES + ONCHAIN_NO_MVRV)]:
            for model_name in ['RandomForest', 'XGBoost']:
                for train_idx, test_idx, fold_name in splits:
                    r = evaluate_one_fold(
                        df, train_idx, test_idx, fold_name,
                        features, label_col, model_name,
                    )
                    results.append(r)
                    if not r.get('skipped'):
                        print(f"  {label_col} | {fs_name} | {model_name} | "
                              f"{fold_name} | macroF1={r['macro_f1']:.3f}")
    with open(os.path.join(MODELS_DIR, 'sensitivity_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"  Saved sensitivity to models/sensitivity_results.json")
    return results


# ============================================================
# Summary tables
# ============================================================
def print_summary(all_results):
    print(f"\n{'='*70}")
    print("  SUMMARY: Mean macro-F1 (averaged over non-trivial folds)")
    print(f"{'='*70}")
    df = pd.DataFrame(all_results)
    valid = df[~df['skipped']].copy() if 'skipped' in df.columns else df
    if valid.empty:
        print("  (no non-skipped folds)")
        return
    summary = (
        valid.groupby(['label', 'fold', 'model'])['macro_f1'].first().unstack('model')
    )
    print(summary.round(3).to_string())

    print(f"\n  -- Mean ± std across folds (per label, per feature set context) --")
    # Per label & model: mean macro F1 across folds (using the summary above's row index)
    mean_f1 = valid.groupby(['label', 'model'])['macro_f1'].agg(['mean', 'std', 'count'])
    print(mean_f1.round(3).to_string())


# ============================================================
# Main entry
# ============================================================
def main():
    print("Loading processed dataset...")
    df = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'btc_features_labels.csv'),
        index_col=0, parse_dates=True,
    )
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    all_results = run_full_experiment(df)
    sens_results = run_sensitivity_analysis(df)

    print_summary(all_results)
    print(f"\nDone. Results in models/results_full.json + models/sensitivity_results.json")


if __name__ == '__main__':
    main()
