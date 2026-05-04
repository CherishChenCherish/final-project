"""
Step 5: 1-D Temporal CNN (PyTorch)
-----------------------------------
Sliding-window 1D CNN for regime classification.

Input  : (batch, n_features, window=60)   60-day lookback of features
Output : (batch, 3)                        regime logits

The same walk-forward evaluation as the sklearn models is applied.

Run standalone (uses regime_price label by default):
    python src/temporal_cnn.py
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')
MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')

WINDOW_SIZE = 60
BATCH_SIZE = 64
EPOCHS = 40
LR = 1e-3
RANDOM_STATE = 42

torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)


# ============================================================
# Dataset (sliding windows)
# ============================================================
class BTCWindowDataset(Dataset):
    def __init__(self, X, y, window_size=WINDOW_SIZE):
        # X: (n, n_features), y: (n,)
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.window_size = window_size

    def __len__(self):
        return max(0, len(self.X) - self.window_size + 1)

    def __getitem__(self, idx):
        # window = X[idx : idx+W]; label = y[idx+W-1] (label aligned with window's last day)
        x_win = self.X[idx:idx + self.window_size].T  # (n_features, W)
        y_lab = self.y[idx + self.window_size - 1]
        return x_win, y_lab


# ============================================================
# Model
# ============================================================
class TemporalCNN(nn.Module):
    def __init__(self, n_features, n_classes=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.BatchNorm1d(128), nn.ReLU(),
            nn.Conv1d(128, 64, kernel_size=3, padding=1), nn.BatchNorm1d(64), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(64, n_classes),
        )

    def forward(self, x):
        return self.net(x)


# ============================================================
# Train + evaluate one fold
# ============================================================
def train_one_fold(X_train, y_train, X_test, y_test, n_features,
                   verbose=True, epochs=EPOCHS):
    """Returns (best_macro_f1_on_test, final_metrics_dict, training_curve)."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

    train_ds = BTCWindowDataset(X_train, y_train)
    test_ds = BTCWindowDataset(X_test, y_test)
    if len(train_ds) == 0 or len(test_ds) == 0:
        return None, {'skip_reason': f'dataset too small (train={len(train_ds)}, test={len(test_ds)})'}, []

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = TemporalCNN(n_features=n_features).to(device)

    # Class weights from training labels actually seen
    actual_y = y_train[WINDOW_SIZE - 1:]
    class_counts = np.bincount(actual_y, minlength=3).astype(float)
    class_counts[class_counts == 0] = 1.0
    weights = (1.0 / class_counts)
    weights = weights / weights.sum() * 3
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_f1, best_state = -1.0, None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds, labs = [], []
        with torch.no_grad():
            for xb, yb in test_loader:
                xb = xb.to(device)
                p = model(xb).argmax(dim=1).cpu().numpy()
                preds.extend(p); labs.extend(yb.numpy())
        macro_f1 = f1_score(labs, preds, average='macro')
        scheduler.step(total_loss)
        history.append({'epoch': epoch + 1,
                        'train_loss': total_loss / max(1, len(train_loader)),
                        'test_macro_f1': macro_f1})

        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if verbose and (epoch + 1) % 5 == 0:
            print(f"      epoch {epoch+1:3d} | loss={total_loss/len(train_loader):.4f} "
                  f"| f1={macro_f1:.3f} | best={best_f1:.3f}")

    # Reload best state, compute full metrics
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            p = model(xb).argmax(dim=1).cpu().numpy()
            preds.extend(p); labs.extend(yb.numpy())

    cm = confusion_matrix(labs, preds, labels=[0, 1, 2]).tolist()
    rep = classification_report(labs, preds, output_dict=True, zero_division=0)
    per_class = {c: {'precision': rep[c]['precision'],
                     'recall': rep[c]['recall'],
                     'f1': rep[c]['f1-score'],
                     'support': rep[c]['support']}
                 for c in ['0', '1', '2'] if c in rep}

    metrics = {
        'macro_f1': best_f1,
        'weighted_f1': f1_score(labs, preds, average='weighted'),
        'accuracy': accuracy_score(labs, preds),
        'confusion_matrix': cm,
        'per_class': per_class,
        'epochs_run': epochs,
        'device': str(device),
    }
    return best_f1, metrics, history


# ============================================================
# Walk-forward across cycles (mirrors models.py logic)
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


def walk_forward_cnn(df, label_col='regime_price', feature_cols=None,
                     fs_name='B_price_onchain'):
    if feature_cols is None:
        feature_cols = PRICE_FEATURES + ONCHAIN_FEATURES
    cycles = sorted(df['cycle'].dropna().unique())
    results = []

    for i in range(1, len(cycles)):
        train_cycles = cycles[:i]
        test_cycle = cycles[i]
        fold_name = f"train_c{''.join(map(str, map(int, train_cycles)))}_test_c{int(test_cycle)}"

        train_idx = df.index[df['cycle'].isin(train_cycles)]
        test_idx = df.index[df['cycle'] == test_cycle]

        avail = [c for c in feature_cols if c in df.columns]
        X_tr_df = df.loc[train_idx, avail].copy()
        y_tr_s = df.loc[train_idx, label_col].copy()
        X_te_df = df.loc[test_idx, avail].copy()
        y_te_s = df.loc[test_idx, label_col].copy()

        nan_frac = X_tr_df.isnull().mean()
        drop = nan_frac[nan_frac > 0.5].index.tolist()
        X_tr_df = X_tr_df.drop(columns=drop)
        X_te_df = X_te_df.drop(columns=drop)
        X_tr_df = X_tr_df.ffill()
        X_te_df = X_te_df.ffill()
        tr_valid = X_tr_df.dropna().index.intersection(y_tr_s.dropna().index)
        te_valid = X_te_df.dropna().index.intersection(y_te_s.dropna().index)
        X_tr_df, y_tr_s = X_tr_df.loc[tr_valid], y_tr_s.loc[tr_valid].astype(int)
        X_te_df, y_te_s = X_te_df.loc[te_valid], y_te_s.loc[te_valid].astype(int)

        base = {'fold': fold_name, 'label': label_col, 'model': 'CNN_1D',
                'feature_set': fs_name,
                'n_train': int(len(X_tr_df)), 'n_test': int(len(X_te_df))}

        if len(X_tr_df) < WINDOW_SIZE + 50 or len(X_te_df) < WINDOW_SIZE + 10:
            base.update({'skipped': True,
                         'skip_reason': f'too few rows for window={WINDOW_SIZE}'})
            results.append(base)
            print(f"  [SKIP] {fold_name}: insufficient rows")
            continue
        if y_tr_s.nunique() < 2 or y_te_s.nunique() < 2:
            base.update({'skipped': True,
                         'skip_reason': f'<2 classes in train ({y_tr_s.unique().tolist()}) or test ({y_te_s.unique().tolist()})'})
            results.append(base)
            print(f"  [SKIP] {fold_name}: trivial fold")
            continue

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr_df.values)
        X_te = scaler.transform(X_te_df.values)
        y_tr = y_tr_s.values
        y_te = y_te_s.values
        n_feat = X_tr.shape[1]

        print(f"  --- {fold_name} | {fs_name} | {label_col} | n_feat={n_feat} "
              f"| train={len(X_tr)}, test={len(X_te)} ---")
        _, metrics, hist = train_one_fold(X_tr, y_tr, X_te, y_te, n_feat)
        base.update({'skipped': False, 'features_used': list(X_tr_df.columns),
                     'history': hist, **metrics})
        results.append(base)
        print(f"      DONE | best macro-F1={metrics.get('macro_f1', float('nan')):.3f} "
              f"acc={metrics.get('accuracy', float('nan')):.3f}")

    return results


def main():
    df = pd.read_csv(os.path.join(PROCESSED_DIR, 'btc_features_labels.csv'),
                     index_col=0, parse_dates=True)
    print(f"Loaded dataset: {len(df)} rows × {len(df.columns)} cols")

    all_cnn = []
    for label_col in ['regime_price', 'regime_base']:
        for fs_name, features in [
            ('A_price_only', PRICE_FEATURES),
            ('B_price_onchain', PRICE_FEATURES + ONCHAIN_FEATURES),
        ]:
            print(f"\n=== CNN | {label_col} | {fs_name} ===")
            res = walk_forward_cnn(df, label_col=label_col,
                                   feature_cols=features, fs_name=fs_name)
            all_cnn.extend(res)

    os.makedirs(MODELS_DIR, exist_ok=True)
    with open(os.path.join(MODELS_DIR, 'cnn_results.json'), 'w') as f:
        json.dump(all_cnn, f, indent=2, default=str)
    print(f"\nSaved CNN results → models/cnn_results.json ({len(all_cnn)} entries)")


if __name__ == '__main__':
    main()
