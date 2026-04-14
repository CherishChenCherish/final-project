"""
Step 6: Visualization & Analysis
- EDA plots
- SHAP feature importance
- Results comparison charts
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
import shap
import os

FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = {'Accumulation': '#2ecc71', 'Expansion': '#3498db', 'Distribution': '#e74c3c'}


def plot_price_with_regimes(df, price_col='Close', regime_col='regime_base'):
    """Plot BTC price with regime labels as background shading."""
    fig, ax = plt.subplots(figsize=(16, 6))

    close = df[price_col].squeeze() if hasattr(df[price_col], 'squeeze') else df[price_col]
    ax.plot(df.index, close, color='black', linewidth=0.8, label='BTC Price')
    ax.set_yscale('log')

    regime_names = {0: 'Accumulation', 1: 'Expansion', 2: 'Distribution'}
    for regime_val, name in regime_names.items():
        mask = df[regime_col] == regime_val
        ax.fill_between(df.index, close.min(), close.max(),
                        where=mask, alpha=0.2, color=COLORS[name], label=name)

    ax.set_title('BTC Price with Market Regime Labels')
    ax.set_ylabel('Price (USD, log scale)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'price_regimes.png'), dpi=150)
    plt.close()
    print("  Saved figures/price_regimes.png")


def plot_mvrv_distribution(df, mvrv_col='CapMVRVCur', regime_col='regime_base'):
    """Plot MVRV distribution by regime."""
    fig, ax = plt.subplots(figsize=(10, 5))
    regime_names = {0: 'Accumulation', 1: 'Expansion', 2: 'Distribution'}
    for regime_val, name in regime_names.items():
        mask = df[regime_col] == regime_val
        vals = df.loc[mask, mvrv_col].dropna()
        ax.hist(vals, bins=50, alpha=0.5, label=name, color=COLORS[name])
    ax.set_xlabel('MVRV Ratio')
    ax.set_ylabel('Count')
    ax.set_title('MVRV Distribution by Regime')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'mvrv_distribution.png'), dpi=150)
    plt.close()
    print("  Saved figures/mvrv_distribution.png")


def plot_feature_correlation(df, features):
    """Plot correlation heatmap of features."""
    available = [f for f in features if f in df.columns]
    corr = df[available].corr()
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='RdBu_r', center=0,
                ax=ax, square=True, linewidths=0.5)
    ax.set_title('Feature Correlation Matrix')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'feature_correlation.png'), dpi=150)
    plt.close()
    print("  Saved figures/feature_correlation.png")


def plot_class_distribution(df, regime_col='regime_base'):
    """Plot class distribution bar chart."""
    fig, ax = plt.subplots(figsize=(8, 5))
    counts = df[regime_col].value_counts().sort_index()
    names = ['Accumulation', 'Expansion', 'Distribution']
    colors = [COLORS[n] for n in names]
    ax.bar(names, counts.values, color=colors)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 10, f'{v}\n({v/len(df)*100:.1f}%)', ha='center')
    ax.set_title('Regime Label Distribution')
    ax.set_ylabel('Number of Days')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'), dpi=150)
    plt.close()
    print("  Saved figures/class_distribution.png")


def plot_results_comparison(results_dict):
    """Plot macro-F1 comparison across models and feature sets."""
    fig, ax = plt.subplots(figsize=(12, 6))
    # Parse results into plottable format
    data = []
    for key, res_list in results_dict.items():
        fs_name, model_name = key.split('__')
        for r in res_list:
            data.append({
                'Feature Set': fs_name,
                'Model': model_name,
                'Fold': r['fold'],
                'Macro F1': r['macro_f1']
            })
    plot_df = pd.DataFrame(data)
    if plot_df.empty:
        return
    sns.barplot(data=plot_df, x='Model', y='Macro F1', hue='Feature Set', ax=ax)
    ax.axhline(y=0.31, color='gray', linestyle='--', label='Random baseline')
    ax.set_title('Model Comparison: Macro-F1 by Feature Set')
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, 'results_comparison.png'), dpi=150)
    plt.close()
    print("  Saved figures/results_comparison.png")


def plot_shap_importance(model, X, feature_names, model_name='XGBoost'):
    """Generate SHAP summary plot for tree-based model."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    fig = plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X, feature_names=feature_names, show=False)
    plt.title(f'SHAP Feature Importance ({model_name})')
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, f'shap_{model_name.lower()}.png'), dpi=150)
    plt.close()
    print(f"  Saved figures/shap_{model_name.lower()}.png")
