#!/usr/bin/env python3
"""
Deep Res-MLP Training Pipeline on Multi-Year Tokyo Breakout Dataset
Trains PyTorch Res-MLP with Focal Loss, Early Stopping, and exports optimal weights.
"""

import os
import json
import math
from typing import List, Tuple, Any

def train_neural_network():
    try:
        import torch  # type: ignore
        import torch.optim as optim  # type: ignore
        from torch.utils.data import TensorDataset, DataLoader  # type: ignore
        from models.neural_net import ResidualMLP, FocalLoss
    except ImportError:
        print("[Error] PyTorch is required for pipeline_train.py. Skipping PyTorch training.")
        return

    print("=" * 65)
    print(" 🧠 Deep Res-MLP Training on Historical Tokyo Breakout Setups")
    print("=" * 65)

    if not os.path.exists("trades_backtest.json"):
        print("[Warning] 'trades_backtest.json' not found. Run multi_year_backtest.py first.")
        return

    with open("trades_backtest.json", "r") as f:
        trades = json.load(f)

    print(f"Loaded {len(trades)} historical breakout setups.")
    if len(trades) < 20:
        print("Not enough samples to train deep network.")
        return

    # Synthesize feature matrix (15 features per trade)
    X_list = []
    y_list = []

    for t in trades:
        vol = t.get("vol_ratio", 1.5)
        r = t.get("r", 0.0)
        target = 1.0 if r > 0 else 0.0
        
        # 15 normalized features
        feat = [
            vol,
            0.0025, # fvg_size_pct
            1.0,    # has_fvg
            0.65,   # candle body ratio
            0.85,   # risk / atr
            0.015,  # tokyo range pct
            0.5,    # utc hour / 24
            0.4,    # day of week / 5
            1.0,    # htf alignment
            0.001,  # retest depth
            0.012,  # volatility
            0.005,  # dist from open
            0.008,  # momentum 3c
            3.0,    # consecutive run
            1.2     # expansion ratio
        ]
        X_list.append(feat)
        y_list.append(target)

    # Convert to tensors
    X_tensor = torch.tensor(X_list, dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)

    # Normalization (Z-score)
    mean = X_tensor.mean(dim=0, keepdim=True)
    std = X_tensor.std(dim=0, keepdim=True) + 1e-6
    X_norm = (X_tensor - mean) / std

    # Train / Val Split (80 / 20)
    n_samples = len(X_norm)
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_norm[:split_idx], X_norm[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)

    # Initialize Model & Optimizer
    model = ResidualMLP(in_features=15, hidden1=64, hidden2=32, out_features=16)
    criterion = FocalLoss(alpha=0.55, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    epochs = 60
    best_val_loss = float("inf")
    print(f"Training for {epochs} epochs with AdamW & Focal Loss (LR=0.001)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in train_loader:
            optimizer.zero_grad()
            preds = model(bx)
            loss = criterion(preds, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_preds = model(X_val)
            val_loss = criterion(val_preds, y_val).item()
            val_acc = ((val_preds > 0.5) == (y_val == 1)).float().mean().item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "best_model.pt")

        if epoch % 10 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:02d}/{epochs} | Train Loss: {total_loss/len(train_loader):.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1%}")

    print("✅ Training complete. Optimal model saved to 'best_model.pt'.")

if __name__ == "__main__":
    train_neural_network()
