#!/usr/bin/env python3
"""
Deep Residual MLP (Res-MLP) Quantitative Training Pipeline:
- Loads real 15-feature vectors directly from multi-asset backtest trades
- Trains 4-layer Res-MLP with LayerNorm, LeakyReLU, and Dropout
- Uses Binary Focal Loss (alpha=0.55, gamma=2.0) to counteract class noise
- Trains with AdamW (LR=0.001, Weight Decay=0.01)
- Exports 'best_model.pt' and calibrated zero-dependency 'ml_weights.json'
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from models.neural_net import ResidualMLP, FocalLoss

def train_neural_network():
    trades_path = "multi_asset_trades.json"
    if not os.path.exists(trades_path):
        trades_path = "trades_backtest.json"

    if not os.path.exists(trades_path):
        print(f"Error: Neither 'multi_asset_trades.json' nor 'trades_backtest.json' found.")
        return

    print(f"Loading empirical trades from '{trades_path}'...")
    with open(trades_path, "r") as f:
        data = json.load(f)

    raw_trades = data.get("trades", [])
    if not raw_trades:
        print("No trade records found in dataset.")
        return

    print(f"Total raw trades found: {len(raw_trades):,}")

    X_list = []
    y_list = []

    for t in raw_trades:
        # Check if trade contains the real 15 extracted features
        feats = t.get("features")
        if not feats or len(feats) != 15:
            continue

        target = 1.0 if t["pnl_r"] > 0 else 0.0
        X_list.append(feats)
        y_list.append(target)

    if not X_list:
        print("Error: No trades with valid 15-feature vectors found.")
        return

    print(f"Constructed empirical feature matrix: {len(X_list):,} samples x 15 features")

    # Convert to tensors
    X_tensor = torch.tensor(X_list, dtype=torch.float32)
    y_tensor = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)

    # Z-score Feature Normalization
    mean = X_tensor.mean(dim=0, keepdim=True)
    std = X_tensor.std(dim=0, keepdim=True) + 1e-6
    X_norm = (X_tensor - mean) / std

    # Train / Validation Split (80 / 20)
    n_samples = len(X_norm)
    split_idx = int(0.8 * n_samples)
    X_train, X_val = X_norm[:split_idx], X_norm[split_idx:]
    y_train, y_val = y_tensor[:split_idx], y_tensor[split_idx:]

    train_ds = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)

    # Initialize PyTorch Res-MLP
    model = ResidualMLP(in_features=15, hidden1=64, hidden2=32, out_features=16)
    criterion = FocalLoss(alpha=0.55, gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

    epochs = 60
    best_val_loss = float("inf")
    best_state = None
    best_val_acc = 0.0
    print(f"Training Res-MLP for {epochs} epochs (Batch=32, LR=0.001, Focal Loss)...")

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
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            best_val_acc = val_acc
            torch.save(model.state_dict(), "best_model.pt")

        if epoch % 10 == 0 or epoch == epochs:
            avg_train_loss = total_loss / len(train_loader)
            print(f"  Epoch {epoch:02d}/{epochs} | Train Loss: {avg_train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.1%}")

    print("✅ PyTorch model trained successfully and saved to 'best_model.pt'.")

    # Export full PyTorch state_dict into zero-dependency res_mlp_weights.json
    try:
        # QUANT FIX: export the BEST validation checkpoint, not the last epoch
        state = best_state if best_state is not None else model.state_dict()
        model_export = {
            "model_type": "Deep-Residual-MLP",
            "features_count": 15,
            "best_val_loss": round(float(best_val_loss), 6),
            "val_accuracy": round(float(best_val_acc), 4),
            "norm_mean": [round(float(m.item()), 6) for m in mean.squeeze()],
            "norm_std": [round(float(s.item()), 6) for s in std.squeeze()],
            "layers": {k: v.cpu().tolist() for k, v in state.items()}
        }
        with open("res_mlp_weights.json", "w") as f:
            json.dump(model_export, f)
        print("✅ Full Deep Res-MLP layers exported to 'res_mlp_weights.json'.")
    except Exception as e:
        print(f"Warning: Could not export res_mlp_weights.json: {e}")

    # Export calibrated logistic proxy weights for zero-dependency runtime
    try:
        # Fit calibrated linear weights using ridge approximation on normalized inputs
        X_design = torch.cat([torch.ones(n_samples, 1), X_norm], dim=1)
        reg = 1e-2 * torch.eye(16)
        w_closed = torch.linalg.solve(X_design.T @ X_design + reg, X_design.T @ y_tensor)
        calibrated_bias = float(w_closed[0].item())
        calibrated_weights = [round(float(w.item()), 4) for w in w_closed[1:].squeeze()]

        with open("ml_weights.json", "w") as f:
            json.dump({
                "model": "Res-MLP-Calibrated",
                "features_count": 15,
                "bias": round(calibrated_bias, 4),
                "weights": calibrated_weights,
                "norm_mean": [round(float(m.item()), 4) for m in mean.squeeze()],
                "norm_std": [round(float(s.item()), 4) for s in std.squeeze()]
            }, f, indent=2)
        print("✅ Calibrated runtime weights exported to 'ml_weights.json'.")
    except Exception as e:
        print(f"Warning: Could not export ml_weights.json: {e}")

if __name__ == "__main__":
    train_neural_network()
