"""
Step 3 & 4: Model Training & Evaluation
- Walk-forward expanding window cross-validation
- Model A (price-only) vs Model B (price + on-chain)
- Ridge, Linear SVM, Random Forest, XGBoost, 1D Temporal CNN
"""

import pandas as pd
import numpy as np
import os
import json
from sklearn.linear_model import RidgeClassifier
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

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

# For MVRV ablation: on-chain features WITHOUT MVRV
ONCHAIN_NO_MVRV = [f for f in ONCHAIN_FEATURES if 'MVRV' not in f]


# ============================================================
# Walk-Forward Expanding Window
# ============================================================
def get_walk_forward_splits(df, cycle_col='cycle'):
    """
    Generate walk-forward train/test splits based on halving cycles.
    Train on cycles 1..k, test on cycle k+1.
    Yields: (train_idx, test_idx, fold_name)
    """
    cycles = sorted(df[cycle_col].dropna().unique())
    for i in range(1, len(cycles)):
        train_cycles = cycles[:i]
        test_cycle = cycles[i]
        train_mask = df[cycle_col].isin(train_cycles)
        test_mask = df[cycle_col] == test_cycle
        yield (
            df.index[train_mask],
            df.index[test_mask],
            f"train_c{''.join(map(str,train_cycles))}_test_c{test_cycle}"
        )


# ============================================================
# Model Definitions
# ============================================================
def get_models():
    """Return dict of model name -> sklearn estimator."""
    return {
        'Ridge': RidgeClassifier(alpha=1.0),
        'LinearSVM': LinearSVC(C=1.0, max_iter=5000, dual='auto'),
        'RandomForest': RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric='mlogloss', random_state=42, n_jobs=-1,
            use_label_encoder=False
        ),
    }


# ============================================================
# Training & Evaluation Pipeline
# ============================================================
def train_and_evaluate(df, feature_cols, label_col='regime_base', model_name='Ridge'):
    """
    Run walk-forward evaluation for one model + one feature set.
    Returns per-fold results.
    """
    models_dict = get_models()
    results = []

    for train_idx, test_idx, fold_name in get_walk_forward_splits(df):
        # Select available features (some on-chain missing in early folds)
        available = [c for c in feature_cols if c in df.columns]
        X_train = df.loc[train_idx, available].copy()
        y_train = df.loc[train_idx, label_col].copy()
        X_test = df.loc[test_idx, available].copy()
        y_test = df.loc[test_idx, label_col].copy()

        # Drop columns that are >50% NaN in training set, then drop remaining NaN rows
        nan_frac = X_train.isnull().mean()
        drop_cols = nan_frac[nan_frac > 0.5].index.tolist()
        if drop_cols:
            X_train = X_train.drop(columns=drop_cols)
            X_test = X_test.drop(columns=drop_cols)

        # Forward-fill then drop remaining NaN rows
        X_train = X_train.ffill()
        X_test = X_test.ffill()
        train_valid = X_train.dropna().index.intersection(y_train.dropna().index)
        test_valid = X_test.dropna().index.intersection(y_test.dropna().index)
        X_train, y_train = X_train.loc[train_valid], y_train.loc[train_valid]
        X_test, y_test = X_test.loc[test_valid], y_test.loc[test_valid]

        if len(X_train) < 50 or len(X_test) < 20:
            print(f"  Skipping {fold_name}: insufficient data (train={len(X_train)}, test={len(X_test)})")
            continue

        # Skip folds where train or test has only 1 class
        if y_train.nunique() < 2:
            print(f"  Skipping {fold_name}: only 1 class in training data")
            continue
        if y_test.nunique() < 2:
            print(f"  Skipping {fold_name}: only 1 class in test data (trivial fold)")
            continue

        # Standardize features (fit on train only)
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train
        model = models_dict[model_name]
        model.fit(X_train_scaled, y_train)

        # Predict
        y_pred = model.predict(X_test_scaled)

        # Evaluate
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        results.append({
            'fold': fold_name,
            'model': model_name,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'macro_f1': macro_f1,
            'report': report,
            'confusion_matrix': cm.tolist(),
            'features_used': available,
        })
        print(f"  {fold_name} | {model_name} | macro-F1={macro_f1:.4f} | "
              f"train={len(X_train)}, test={len(X_test)}")

    return results


# ============================================================
# Full Experiment: A vs B comparison
# ============================================================
def run_full_experiment(df, label_col='regime_base'):
    """
    Run all models with:
    - Feature set A: price-only
    - Feature set B: price + on-chain
    - Feature set C: price + on-chain (no MVRV) [ablation]
    """
    os.makedirs(MODELS_DIR, exist_ok=True)
    all_results = {}

    feature_sets = {
        'A_price_only': PRICE_FEATURES,
        'B_price_onchain': PRICE_FEATURES + ONCHAIN_FEATURES,
        'C_ablation_no_mvrv': PRICE_FEATURES + ONCHAIN_NO_MVRV,
    }

    for fs_name, features in feature_sets.items():
        print(f"\n{'='*60}")
        print(f"Feature Set: {fs_name}")
        print(f"{'='*60}")
        for model_name in get_models():
            print(f"\n--- {model_name} ---")
            results = train_and_evaluate(df, features, label_col, model_name)
            all_results[f"{fs_name}__{model_name}"] = results

    # Save results
    # (Convert to serializable format)
    save_results = {}
    for key, res_list in all_results.items():
        save_results[key] = [{
            k: v for k, v in r.items() if k != 'report'
        } for r in res_list]

    with open(os.path.join(MODELS_DIR, 'results.json'), 'w') as f:
        json.dump(save_results, f, indent=2)

    return all_results


# ============================================================
# Sensitivity Analysis across threshold variants
# ============================================================
def run_sensitivity_analysis(df):
    """Run best model across all three label threshold variants."""
    variants = ['regime_base', 'regime_conservative', 'regime_aggressive']
    for variant in variants:
        print(f"\n{'='*60}")
        print(f"Sensitivity: {variant}")
        print(f"{'='*60}")
        features = PRICE_FEATURES + ONCHAIN_FEATURES
        for model_name in ['RandomForest', 'XGBoost']:
            results = train_and_evaluate(df, features, variant, model_name)


# ============================================================
# Main
# ============================================================
def main():
    print("Loading processed dataset...")
    df = pd.read_csv(
        os.path.join(PROCESSED_DIR, 'btc_features_labels.csv'),
        index_col=0, parse_dates=True
    )
    print(f"  {len(df)} rows, {len(df.columns)} columns")

    # Run main experiment
    results = run_full_experiment(df)

    # Print summary: A vs B delta
    print("\n" + "="*60)
    print("SUMMARY: On-chain Marginal Value (F1 delta B - A)")
    print("="*60)
    for model_name in get_models():
        a_key = f"A_price_only__{model_name}"
        b_key = f"B_price_onchain__{model_name}"
        if a_key in results and b_key in results:
            for a_res, b_res in zip(results[a_key], results[b_key]):
                delta = b_res['macro_f1'] - a_res['macro_f1']
                print(f"  {model_name} | {a_res['fold']} | "
                      f"A={a_res['macro_f1']:.4f} B={b_res['macro_f1']:.4f} "
                      f"delta={delta:+.4f}")

    # Sensitivity analysis
    run_sensitivity_analysis(df)


if __name__ == '__main__':
    main()
