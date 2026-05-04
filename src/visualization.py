"""
Step 6: Visualization
---------------------
All plots needed for the final report.

Run: python src/visualization.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # no display server (headless)
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_DIR = os.path.join(ROOT, 'data', 'processed')
MODELS_DIR    = os.path.join(ROOT, 'models')
FIGURES_DIR   = os.path.join(ROOT, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {
    'Bear/Accum':     '#2ecc71',
    'Sideways/Exp':   '#3498db',
    'Bull/Distrib':   '#e74c3c',
}
LABEL_NAMES = {0: 'Bear/Accum', 1: 'Sideways/Exp', 2: 'Bull/Distrib'}


# ============================================================
# Load all results
# ============================================================
def load_results():
    with open(os.path.join(MODELS_DIR, 'results_full.json')) as f:
        full = json.load(f)
    with open(os.path.join(MODELS_DIR, 'sensitivity_results.json')) as f:
        sens = json.load(f)
    with open(os.path.join(MODELS_DIR, 'cnn_results.json')) as f:
        cnn = json.load(f)
    return full, sens, cnn


# ============================================================
# 1. Price + regime background
# ============================================================
def plot_price_with_regimes(df, regime_col='regime_price', save_name=None):
    fig, ax = plt.subplots(figsize=(15, 5))
    close = df['Close']
    ax.plot(df.index, close, color='black', linewidth=0.7)
    ax.set_yscale('log')
    for k, name in LABEL_NAMES.items():
        mask = df[regime_col] == k
        ax.fill_between(df.index, close.min(), close.max(),
                        where=mask, alpha=0.18, color=COLORS[name], label=name)
    halvings = ['2016-07-09', '2020-05-11', '2024-04-20']
    for h in halvings:
        ax.axvline(pd.Timestamp(h), color='gray', linestyle='--', alpha=0.5)
    ax.set_title(f'BTC price with regime labels ({regime_col})')
    ax.set_ylabel('Price USD (log)')
    ax.legend(loc='upper left', framealpha=0.9)
    fig.tight_layout()
    save_name = save_name or f'price_regimes_{regime_col}.png'
    fig.savefig(os.path.join(FIGURES_DIR, save_name), dpi=140)
    plt.close()
    print(f"  saved {save_name}")


# ============================================================
# 2. Class distribution per label set
# ============================================================
def plot_class_distributions(df):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, col in zip(axes, ['regime_base', 'regime_conservative', 'regime_aggressive', 'regime_price']):
        counts = df[col].value_counts().sort_index()
        names = [LABEL_NAMES[int(i)] for i in counts.index if pd.notna(i)]
        cols = [COLORS[n] for n in names]
        ax.bar(names, counts.values, color=cols)
        for i, v in enumerate(counts.values):
            pct = v / counts.sum() * 100
            ax.text(i, v + max(counts.values) * 0.01, f'{v}\n({pct:.0f}%)',
                    ha='center', fontsize=9)
        ax.set_title(col)
        ax.tick_params(axis='x', rotation=20)
    fig.suptitle('Label distributions across all four label-construction schemes')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'), dpi=140)
    plt.close()
    print("  saved class_distribution.png")


# ============================================================
# 3. MVRV histogram by regime
# ============================================================
def plot_mvrv_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    for k, name in LABEL_NAMES.items():
        vals = df.loc[df['regime_base'] == k, 'CapMVRVCur'].dropna()
        ax.hist(vals, bins=60, alpha=0.55, label=name, color=COLORS[name])
    for x, txt in [(1.0, 'base accum'), (3.5, 'base distrib')]:
        ax.axvline(x, color='black', linestyle='--', alpha=0.6)
        ax.text(x, ax.get_ylim()[1] * 0.95, f' MVRV={x}\n {txt}', fontsize=8, va='top')
    ax.set_xlabel('MVRV ratio')
    ax.set_ylabel('Days')
    ax.set_title('MVRV distribution by regime_base label')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'mvrv_distribution.png'), dpi=140)
    plt.close()
    print("  saved mvrv_distribution.png")


# ============================================================
# 4. Feature correlation heatmap
# ============================================================
def plot_feature_correlation(df):
    feats = [
        'SMA_30', 'Price_SMA_Ratio', 'RSI_14', 'MACD', 'Volatility_30d', 'Returns',
        'CapMVRVCur', 'MVRV_Z', 'Puell_Multiple',
        'AdrActCnt_log', 'HashRate_log', 'TxCnt_log',
        'NetFlowExUSD', 'ExchangeSupplyRatio',
    ]
    avail = [f for f in feats if f in df.columns]
    corr = df[avail].corr()
    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, square=True, linewidths=0.4, cbar_kws={'shrink': 0.7})
    ax.set_title('Feature correlation matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'feature_correlation.png'), dpi=140)
    plt.close()
    print("  saved feature_correlation.png")


# ============================================================
# 5. Main results: per-fold macro F1, A vs B vs C, both labels
# ============================================================
def parse_to_df(records):
    """Flatten list-of-dicts to DataFrame."""
    rows = []
    for r in records:
        if r.get('skipped'):
            rows.append({**r, 'macro_f1': np.nan})
        else:
            rows.append(r)
    return pd.DataFrame(rows)


def plot_results_main(full):
    """Bar chart: macro-F1 per (model, feature_set, fold), faceted by label."""
    df = parse_to_df(full).copy()
    # Add feature_set column from the iteration order in run_full_experiment
    # The records don't carry feature_set directly — infer from features_used
    def infer_fs(features_used):
        if not isinstance(features_used, list):
            return 'unknown'
        feats = set(features_used)
        has_mvrv = any('MVRV' in f for f in feats)
        has_onchain = any(f in feats for f in
                          ['AdrActCnt', 'HashRate', 'Puell_Multiple',
                           'AdrActCnt_log', 'HashRate_log'])
        if not has_onchain:
            return 'A_price_only'
        if has_mvrv:
            return 'B_price_onchain'
        return 'C_no_mvrv_ablation'
    df['feature_set'] = df['features_used'].apply(infer_fs)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    for ax, label in zip(axes, ['regime_base', 'regime_price']):
        sub = df[(df['label'] == label) & (~df['macro_f1'].isna())]
        if sub.empty:
            ax.text(0.5, 0.5, 'no valid folds', transform=ax.transAxes, ha='center')
            continue
        sns.barplot(data=sub, x='model', y='macro_f1',
                    hue='feature_set', ax=ax,
                    order=['Majority', 'Ridge', 'LinearSVM', 'RandomForest', 'XGBoost'],
                    hue_order=['A_price_only', 'B_price_onchain', 'C_no_mvrv_ablation'],
                    errorbar='sd')
        ax.set_title(f'Label = {label}')
        ax.set_ylabel('macro-F1 (mean across folds, ±1 sd)')
        ax.set_xlabel('')
        ax.legend(title='', loc='upper left', fontsize=8)
        ax.set_ylim(0, 1.05)
    fig.suptitle('Walk-forward macro-F1 by model × feature set × label')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'results_main.png'), dpi=140)
    plt.close()
    print("  saved results_main.png")
    return df


# ============================================================
# 6. MVRV leakage demonstration
# ============================================================
def plot_leakage_demo(full_df):
    """Show: regime_base + B (with MVRV) inflates F1 vs C (no MVRV)."""
    sub = full_df[full_df['label'] == 'regime_base'].copy()
    sub = sub[~sub['macro_f1'].isna()]
    if sub.empty:
        print("  (no data for leakage demo)")
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    grp = sub.groupby(['model', 'feature_set'])['macro_f1'].mean().unstack()
    grp = grp.reindex(['Majority', 'Ridge', 'LinearSVM', 'RandomForest', 'XGBoost'])
    grp = grp[['A_price_only', 'B_price_onchain', 'C_no_mvrv_ablation']]
    grp.plot(kind='bar', ax=ax, color=['#888888', '#e74c3c', '#3498db'], width=0.8)
    ax.set_title('regime_base label — B (with MVRV) inflates F1 vs C (no MVRV)\n'
                 'Difference between B and C ≈ size of MVRV self-leakage')
    ax.set_ylabel('mean macro-F1 across folds')
    ax.set_xlabel('')
    ax.set_ylim(0, 1.0)
    ax.legend(title='feature set', loc='upper left')
    plt.xticks(rotation=15)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'leakage_demo.png'), dpi=140)
    plt.close()
    print("  saved leakage_demo.png")


# ============================================================
# 7. Confusion matrices (best models)
# ============================================================
def plot_confusion_matrices(full):
    """Pick: best XGBoost on regime_price fold 3, best CNN on regime_price fold 3, etc."""
    # Show 4 confusion matrices in a grid
    interesting = []
    for r in full:
        if r.get('skipped'): continue
        if (r['label'] == 'regime_price' and
                r['model'] in ['XGBoost', 'RandomForest']):
            if 'features_used' in r and any('MVRV' in f for f in r['features_used']):
                interesting.append(('B+regime_price', r))
        if (r['label'] == 'regime_base' and
                r['model'] == 'XGBoost' and
                'features_used' in r and any('MVRV' in f for f in r['features_used'])):
            interesting.append(('B+regime_base (LEAKY)', r))

    # Take up to 4
    show = interesting[:4]
    if not show:
        return
    fig, axes = plt.subplots(1, len(show), figsize=(4.5 * len(show), 4))
    if len(show) == 1: axes = [axes]
    for ax, (tag, r) in zip(axes, show):
        cm = np.array(r['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Bear/Acc', 'Side/Exp', 'Bull/Dist'],
                    yticklabels=['Bear/Acc', 'Side/Exp', 'Bull/Dist'])
        ax.set_title(f"{tag}\n{r['model']} {r['fold']}\nF1={r.get('macro_f1', 0):.2f}", fontsize=9)
        ax.set_xlabel('predicted'); ax.set_ylabel('true')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'confusion_matrices.png'), dpi=140)
    plt.close()
    print("  saved confusion_matrices.png")


# ============================================================
# 8. CNN: training curves
# ============================================================
def plot_cnn_training(cnn_results):
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    axes = axes.flatten()
    plotted = 0
    for r in cnn_results:
        if r.get('skipped'): continue
        if 'history' not in r or not r['history']: continue
        if plotted >= 4: break
        ax = axes[plotted]
        hist = pd.DataFrame(r['history'])
        ax2 = ax.twinx()
        ax.plot(hist['epoch'], hist['train_loss'], color='steelblue', label='train loss')
        ax2.plot(hist['epoch'], hist['test_macro_f1'], color='crimson', label='test macro-F1')
        ax.set_title(f"CNN | {r['label']} | {r['feature_set']}\n{r['fold']} | "
                     f"best F1={r.get('macro_f1', 0):.3f}", fontsize=10)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss', color='steelblue')
        ax2.set_ylabel('macro-F1', color='crimson')
        ax.tick_params(axis='y', labelcolor='steelblue')
        ax2.tick_params(axis='y', labelcolor='crimson')
        plotted += 1
    for j in range(plotted, 4):
        axes[j].axis('off')
    fig.suptitle('1D Temporal CNN training dynamics')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'cnn_training.png'), dpi=140)
    plt.close()
    print("  saved cnn_training.png")


# ============================================================
# 9. CNN vs sklearn models
# ============================================================
def plot_cnn_vs_sklearn(full_df, cnn_records):
    """Bar chart: CNN vs others on regime_price."""
    rows = []
    for r in full_df.to_dict('records'):
        if r.get('label') == 'regime_price' and not r.get('skipped'):
            rows.append({'model': r['model'], 'fold': r['fold'],
                         'feature_set': r['feature_set'],
                         'macro_f1': r['macro_f1']})
    for r in cnn_records:
        if r.get('skipped') or r.get('label') != 'regime_price': continue
        rows.append({'model': 'CNN_1D', 'fold': r['fold'],
                     'feature_set': r['feature_set'],
                     'macro_f1': r.get('macro_f1', np.nan)})
    if not rows:
        return
    plot_df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(11, 5))
    sub = plot_df[plot_df['feature_set'] == 'B_price_onchain']
    sns.barplot(data=sub, x='fold', y='macro_f1', hue='model', ax=ax,
                hue_order=['Majority', 'Ridge', 'LinearSVM', 'RandomForest', 'XGBoost', 'CNN_1D'])
    ax.set_title('regime_price | feature set B (price + on-chain) — CNN vs sklearn models')
    ax.set_ylabel('macro-F1')
    ax.set_ylim(0, 0.6)
    ax.legend(title='', ncol=3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'cnn_vs_sklearn.png'), dpi=140)
    plt.close()
    print("  saved cnn_vs_sklearn.png")


# ============================================================
# 10. SHAP for best XGBoost (regime_price + B feature set, fold 3)
# ============================================================
def plot_shap(df):
    try:
        import shap
        from sklearn.preprocessing import StandardScaler
        from xgboost import XGBClassifier
    except ImportError:
        print("  shap or xgboost missing, skipping SHAP plot")
        return

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
    feats = PRICE_FEATURES + ONCHAIN_FEATURES
    feats = [f for f in feats if f in df.columns]

    train_idx = df.index[df['cycle'].isin([1, 2, 3])]
    test_idx = df.index[df['cycle'] == 4]
    X_tr = df.loc[train_idx, feats].ffill()
    y_tr = df.loc[train_idx, 'regime_price']
    X_te = df.loc[test_idx, feats].ffill()
    valid_tr = X_tr.dropna().index.intersection(y_tr.dropna().index)
    X_tr, y_tr = X_tr.loc[valid_tr], y_tr.loc[valid_tr].astype(int)
    valid_te = X_te.dropna().index
    X_te = X_te.loc[valid_te]
    if X_tr.empty or X_te.empty:
        print("  not enough data for SHAP plot")
        return

    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_tr)
    X_te_s = scaler.transform(X_te)

    model = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05,
                          random_state=42, n_jobs=-1, eval_metric='mlogloss')
    model.fit(X_tr_s, y_tr)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_te_s)

    # XGBoost multi-class returns list-of-arrays, one per class
    # For summary plot, use aggregate (mean abs across classes)
    fig = plt.figure(figsize=(10, 8))
    if isinstance(shap_values, list):
        # Aggregate across classes
        agg = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        shap.summary_plot(agg, X_te_s, feature_names=X_tr.columns.tolist(),
                          plot_type='bar', show=False)
    else:
        shap.summary_plot(shap_values, X_te_s, feature_names=X_tr.columns.tolist(),
                          show=False)
    plt.title('SHAP feature importance — XGBoost on regime_price (cycle 4 test)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, 'shap_xgboost.png'), dpi=140)
    plt.close()
    print("  saved shap_xgboost.png")


# ============================================================
# 11. Sensitivity analysis chart
# ============================================================
def plot_sensitivity(sens):
    df = pd.DataFrame([r for r in sens if not r.get('skipped')])
    if df.empty:
        print("  no sensitivity data")
        return

    def infer_fs(features_used):
        if not isinstance(features_used, list):
            return 'unknown'
        return 'B_price_onchain' if any('MVRV' in f for f in features_used) else 'C_no_mvrv_ablation'
    df['feature_set'] = df['features_used'].apply(infer_fs)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, label in zip(axes, ['regime_conservative', 'regime_aggressive']):
        sub = df[df['label'] == label]
        if sub.empty:
            continue
        sns.barplot(data=sub, x='model', y='macro_f1', hue='feature_set', ax=ax,
                    hue_order=['B_price_onchain', 'C_no_mvrv_ablation'],
                    errorbar='sd')
        ax.set_title(f'{label}')
        ax.set_ylabel('macro-F1')
        ax.set_ylim(0, 1.05)
        ax.legend(title='', loc='upper left')
    fig.suptitle('Sensitivity: alternative MVRV thresholds — gap B vs C measures leakage')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'sensitivity_analysis.png'), dpi=140)
    plt.close()
    print("  saved sensitivity_analysis.png")


# ============================================================
# Main
# ============================================================
def main():
    print("Loading data + results...")
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'btc_features_labels.csv'),
                     index_col=0, parse_dates=True)
    full, sens, cnn = load_results()
    print(f"  data: {len(df)} rows | full: {len(full)} | sens: {len(sens)} | cnn: {len(cnn)}")

    print("\nGenerating figures...")
    plot_price_with_regimes(df, regime_col='regime_price', save_name='price_regimes_price.png')
    plot_price_with_regimes(df, regime_col='regime_base', save_name='price_regimes_base.png')
    plot_class_distributions(df)
    plot_mvrv_distribution(df)
    plot_feature_correlation(df)
    full_df = plot_results_main(full)
    plot_leakage_demo(full_df)
    plot_confusion_matrices(full)
    plot_cnn_training(cnn)
    plot_cnn_vs_sklearn(full_df, cnn)
    plot_sensitivity(sens)
    plot_shap(df)

    print("\nAll figures saved to figures/")


if __name__ == '__main__':
    main()
