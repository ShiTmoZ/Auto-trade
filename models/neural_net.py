"""
Deep Residual Multi-Layer Perceptron (Res-MLP) Architecture for Breakout Validation
Engineered with Focal Loss & Layer Normalization to resist market stochastic noise.
"""

from typing import Any

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
    base_class = nn.Module
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    nn = None
    base_class = object

class ResidualMLP(base_class):
    def __init__(self, in_features: int = 15, hidden1: int = 64, hidden2: int = 32, out_features: int = 16):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for ResidualMLP. Install with 'pip install torch'.")
        super().__init__()
        # Layer 1
        self.fc1 = nn.Linear(in_features, hidden1)
        self.ln1 = nn.LayerNorm(hidden1)
        self.drop1 = nn.Dropout(0.3)
        
        # Layer 2
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.ln2 = nn.LayerNorm(hidden2)
        self.drop2 = nn.Dropout(0.2)
        
        # Skip / Residual projection from Layer 1 to Layer 3
        self.skip_proj = nn.Linear(hidden1, out_features)
        self.fc3 = nn.Linear(hidden2, out_features)
        
        # Output head
        self.head = nn.Linear(out_features, 1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.1)

    def forward(self, x: Any) -> Any:
        # Block 1
        out1 = self.drop1(self.leaky_relu(self.ln1(self.fc1(x))))
        
        # Block 2
        out2 = self.drop2(self.leaky_relu(self.ln2(self.fc2(out1))))
        
        # Residual Fusion: merge processed Block 2 with projected Block 1
        res = self.fc3(out2) + self.skip_proj(out1)
        res = self.leaky_relu(res)
        
        # Probability head (Sigmoid)
        prob = torch.sigmoid(self.head(res))
        return prob

class FocalLoss(base_class):
    """
    Focal Loss for addressing class imbalance and focusing on hard-to-predict setups.
    FL(p_t) = -alpha * (1 - p_t)^gamma * log(p_t)
    """
    def __init__(self, alpha: float = 0.55, gamma: float = 2.0):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for FocalLoss.")
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, preds: Any, targets: Any) -> Any:
        eps = 1e-7
        preds = torch.clamp(preds, eps, 1.0 - eps)
        targets = targets.view(-1, 1)

        # Binary focal loss
        pt = torch.where(targets == 1, preds, 1 - preds)
        alpha_factor = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_weight = alpha_factor * ((1 - pt) ** self.gamma)
        
        loss = -focal_weight * torch.log(pt)
        return loss.mean()
