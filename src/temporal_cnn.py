"""
Step 5: 1-D Temporal CNN (PyTorch)
- Takes a window of past N days as input
- Classifies the regime for the next 30 days
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

WINDOW_SIZE = 60  # 60 days lookback
BATCH_SIZE = 32
EPOCHS = 50
LR = 1e-3


# ============================================================
# Dataset
# ============================================================
class BTCWindowDataset(Dataset):
    """Sliding window dataset for temporal CNN."""

    def __init__(self, X, y, window_size=WINDOW_SIZE):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.window_size = window_size

    def __len__(self):
        return len(self.X) - self.window_size

    def __getitem__(self, idx):
        # Shape: (window_size, n_features) -> transpose to (n_features, window_size) for Conv1d
        x_window = self.X[idx:idx + self.window_size].T
        y_label = self.y[idx + self.window_size - 1]
        return x_window, y_label


# ============================================================
# Model
# ============================================================
class TemporalCNN(nn.Module):
    """1-D Temporal CNN for regime classification."""

    def __init__(self, n_features, n_classes=3):
        super().__init__()
        self.conv1 = nn.Conv1d(n_features, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(64, n_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        # x shape: (batch, n_features, window_size)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)  # (batch, 64)
        x = self.dropout(x)
        x = self.fc(x)
        return x


# ============================================================
# Training Loop
# ============================================================
def train_temporal_cnn(X_train, y_train, X_test, y_test, n_features):
    """Train and evaluate the temporal CNN."""
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"  Using device: {device}")

    # Create datasets
    train_dataset = BTCWindowDataset(X_train, y_train)
    test_dataset = BTCWindowDataset(X_test, y_test)

    if len(train_dataset) == 0 or len(test_dataset) == 0:
        print("  Insufficient data for CNN with current window size")
        return None

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Model
    model = TemporalCNN(n_features=n_features).to(device)

    # Class weights for imbalanced data
    class_counts = np.bincount(y_train[WINDOW_SIZE:])
    if len(class_counts) < 3:
        class_counts = np.append(class_counts, [1] * (3 - len(class_counts)))
    weights = 1.0 / class_counts.astype(float)
    weights = weights / weights.sum() * 3
    class_weights = torch.FloatTensor(weights).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training
    best_f1 = 0
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Evaluate
        model.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.to(device)
                output = model(X_batch)
                preds = output.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(y_batch.numpy())

        macro_f1 = f1_score(all_labels, all_preds, average='macro')
        scheduler.step(total_loss)

        if macro_f1 > best_f1:
            best_f1 = macro_f1

        if (epoch + 1) % 10 == 0:
            print(f"    Epoch {epoch+1}/{EPOCHS} | Loss={total_loss/len(train_loader):.4f} | "
                  f"F1={macro_f1:.4f} | Best={best_f1:.4f}")

    return best_f1


if __name__ == '__main__':
    print("Temporal CNN - run from notebooks or via models.py integration")
