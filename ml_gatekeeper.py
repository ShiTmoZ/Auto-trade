"""
Machine Learning Gatekeeper:
- 15-dimensional Quantitative Feature Vector
- Conservative Learning Rate (0.015) & L2 Regularization to resist noise
- Online Gradient Update & Inference without heavy ML dependencies
"""

import os
import math
import json
from typing import Dict, Any, List, Tuple, Optional

class MLGatekeeper:
    # NOTE: order below MUST mirror extract_15_features() in multi_asset_pipeline.py
    FEATURE_NAMES = [
        "vol_ratio",                 # 0: min(5.0, volume ratio vs 20-SMA)
        "has_fvg",                   # 1: Binary FVG flag (0 or 1)
        "candle0_body_ratio",        # 2: Body / Total Candle Range (c0)
        "tokyo_range_pct",           # 3: Tokyo Range / Price
        "atr_close_ratio",           # 4: ATR14 / Close
        "hour_utc",                  # 5: Normalized UTC hour (0 to 1)
        "day_of_week",               # 6: Normalized day of week (0 to 1)
        "htf_trend_side",            # 7: 1.0 if close above 4H EMA50, else 0.0
        "htf_trend_distance",        # 8: |close - EMA50_4H| / close
        "rsi14_norm",                # 9: (RSI14 - 50) / 50
        "candle1_body_ratio",        # 10: Body / Total Candle Range (c1)
        "vol_ratio_scaled",          # 11: vol_ratio / 3.0
        "range_expansion_ratio",     # 12: Break candle range vs ATR14
        "candle0_bullish",           # 13: 1.0 if c0 close > open, else 0.0
        "constant_bias",             # 14: Hardcoded 0.5 constant (training parity)
    ]

    def __init__(
        self,
        weights_file: str = "ml_weights.json",
        learning_rate: float = 0.015,
        l2_lambda: float = 1.0,
        confidence_threshold: float = 0.65
    ):
        self.weights_file = weights_file
        self.learning_rate = learning_rate
        self.l2_lambda = l2_lambda
        self.confidence_threshold = confidence_threshold
        
        self.weights: List[float] = []
        self.bias: float = 0.0
        self.total_trained_samples: int = 0
        self.norm_mean: Optional[List[float]] = None
        self.norm_std: Optional[List[float]] = None
        self.res_mlp_layers: Optional[Dict[str, Any]] = None
        self._load_or_initialize_weights()

    def _load_or_initialize_weights(self):
        """Loads weights from disk or initializes with robust quantitative defaults."""
        res_mlp_path = os.path.join(os.path.dirname(self.weights_file), "res_mlp_weights.json")
        if os.path.exists(res_mlp_path):
            try:
                with open(res_mlp_path, "r") as f:
                    mlp_data = json.load(f)
                    self.res_mlp_layers = mlp_data.get("layers")
                    self.norm_mean = mlp_data.get("norm_mean")
                    self.norm_std = mlp_data.get("norm_std")
                    print("[MLGatekeeper] 🧠 Loaded Genuine PyTorch Deep Residual MLP layers from res_mlp_weights.json!")
            except Exception as e:
                print(f"[MLGatekeeper] Warning: Could not read {res_mlp_path}: {e}")

        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "r") as f:
                    data = json.load(f)
                    self.weights = data.get("weights", [])
                    self.bias = data.get("bias", 0.0)
                    # QUANT FIX: never clobber Res-MLP calibration stats with missing keys
                    if data.get("norm_mean") and data.get("norm_std"):
                        self.norm_mean = data.get("norm_mean")
                        self.norm_std = data.get("norm_std")
                    self.confidence_threshold = data.get("confidence_threshold", self.confidence_threshold)
                    self.total_trained_samples = data.get("samples", 0)
                    if len(self.weights) == len(self.FEATURE_NAMES):
                        return
            except Exception as e:
                print(f"[MLGatekeeper] Warning: Could not read {self.weights_file}: {e}")

        # Baseline empirical priors (positive weights for volume, FVG, HTF alignment)
        self.weights = [
            0.65,  # vol_ratio (higher volume -> higher win prob)
            0.45,  # fvg_size_pct
            0.50,  # has_fvg (strong institutional signal)
            0.35,  # candle0_body_ratio (strong body > wicks)
            -0.30, # risk_atr_ratio (too wide SL degrades edge)
            -0.20, # tokyo_range_pct (too wide Asian range exhausts movement)
            0.25,  # hour_utc (London/NY overlap preferred)
            0.10,  # day_of_week
            0.75,  # htf_alignment_strength (crucial filter)
            -0.15, # retest_depth_pct
            0.20,  # volatility_ratio
            0.15,  # dist_from_tokyo_open_pct
            0.30,  # momentum_3c_pct
            -0.25, # consecutive_run (extended runs risk exhaustion)
            0.40   # range_expansion_ratio
        ]
        self.bias = 0.15
        self.total_trained_samples = 50 # Baseline prior count
        self._save_weights()

    def _save_weights(self):
        try:
            payload = {
                "weights": self.weights,
                "bias": self.bias,
                "samples": self.total_trained_samples,
                "learning_rate": self.learning_rate,
                "l2_lambda": self.l2_lambda
            }
            # QUANT FIX: persist calibration stats so a restart never loses normalization
            if self.norm_mean and self.norm_std:
                payload["norm_mean"] = self.norm_mean
                payload["norm_std"] = self.norm_std
                payload["confidence_threshold"] = self.confidence_threshold
            with open(self.weights_file, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as e:
            print(f"[MLGatekeeper] Could not save weights: {e}")

    def extract_features(
        self,
        signal: Dict[str, Any],
        tokyo_data: Dict[str, Any],
        htf_data: Dict[str, Any],
        recent_candles: List[Dict[str, Any]]
    ) -> List[float]:
        """
        Unified 15-Feature Quantitative Vector matching multi_asset_pipeline.py & ml_weights.json exactly.
        """
        c0 = recent_candles[-2] if len(recent_candles) >= 2 and not recent_candles[-1].get("is_closed", True) else recent_candles[-1]
        c1 = recent_candles[-1]

        c0_close = max(1.0, float(c0.get("close", 1.0)))
        tokyo_range = abs(float(tokyo_data.get("high", c0_close)) - float(tokyo_data.get("low", c0_close)))
        vol_ratio = float(signal.get("vol_ratio", 1.0))
        has_fvg = bool(signal.get("has_fvg", False))

        ema4h = float(htf_data.get("ema50_4h", c0_close))
        trend_dist = c0_close - ema4h

        rsi_val = float(c0.get("rsi14", 50.0))
        if rsi_val == 50.0 and len(recent_candles) >= 15:
            diffs = [recent_candles[k]["close"] - recent_candles[k-1]["close"] for k in range(-14, 0)]
            gains = [d for d in diffs if d > 0]
            losses = [-d for d in diffs if d < 0]
            avg_gain = sum(gains) / 14.0 if gains else 1e-6
            avg_loss = sum(losses) / 14.0 if losses else 1e-6
            rs = avg_gain / avg_loss
            rsi_val = 100.0 - (100.0 / (1.0 + rs))

        body_ratio = abs(c0["close"] - c0["open"]) / max(0.01, c0["high"] - c0["low"])
        range_expansion = (c0["high"] - c0["low"]) / max(1e-5, c0.get("atr14", 1.0))
        c1_body_ratio = abs(c1["close"] - c1["open"]) / max(0.01, c1["high"] - c1["low"])

        utc_dt = c0["utc_dt"]

        return [
            min(5.0, vol_ratio),
            1.0 if has_fvg else 0.0,
            body_ratio,
            tokyo_range / c0_close,
            c0.get("atr14", 1.0) / c0_close,
            float(utc_dt.hour) / 24.0,
            float(utc_dt.weekday()) / 6.0,
            1.0 if trend_dist > 0 else 0.0,
            abs(trend_dist) / c0_close,
            (rsi_val - 50.0) / 50.0,
            c1_body_ratio,
            vol_ratio / 3.0,
            range_expansion,
            1.0 if c0["close"] > c0["open"] else 0.0,
            0.5
        ]

    def _evaluate_res_mlp(self, x_norm: List[float]) -> float:
        layers = self.res_mlp_layers
        def linear(x, w, b):
            return [sum(x[j] * w[i][j] for j in range(len(x))) + b[i] for i in range(len(w))]

        def layer_norm(x, weight, bias, eps=1e-5):
            n = len(x)
            mean = sum(x) / n
            var = sum((xi - mean) ** 2 for xi in x) / n
            std = math.sqrt(var + eps)
            return [((x[i] - mean) / std) * weight[i] + bias[i] for i in range(n)]

        def leaky_relu(x, negative_slope=0.1):
            return [xi if xi > 0 else xi * negative_slope for xi in x]

        # Block 1
        h1 = linear(x_norm, layers["fc1.weight"], layers["fc1.bias"])
        ln1 = layer_norm(h1, layers["ln1.weight"], layers["ln1.bias"])
        out1 = leaky_relu(ln1)

        # Block 2
        h2 = linear(out1, layers["fc2.weight"], layers["fc2.bias"])
        ln2 = layer_norm(h2, layers["ln2.weight"], layers["ln2.bias"])
        out2 = leaky_relu(ln2)

        # Skip Projection & Fusion
        proj = linear(out1, layers["skip_proj.weight"], layers["skip_proj.bias"])
        fc3 = linear(out2, layers["fc3.weight"], layers["fc3.bias"])
        res = leaky_relu([fc3[i] + proj[i] for i in range(len(fc3))])

        # Probability Head
        logit = linear(res, layers["head.weight"], layers["head.bias"])[0]
        prob = 1.0 / (1.0 + math.exp(-max(-20.0, min(20.0, logit))))
        return prob

    def evaluate_signal(self, features: List[float]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Calculates win probability using the Deep Residual MLP or regularized linear model.
        Returns: (is_approved, confidence_score, explanation_dict)
        """
        if len(features) != len(self.weights):
            return False, 0.0, {"error": "Feature count mismatch"}

        # Normalize features if calibration statistics are loaded
        if self.norm_mean and self.norm_std and len(self.norm_mean) == len(features):
            norm_feats = [(f - m) / (s + 1e-6) for f, m, s in zip(features, self.norm_mean, self.norm_std)]
        else:
            norm_feats = features

        if self.res_mlp_layers:
            prob = self._evaluate_res_mlp(norm_feats)
        else:
            # Fallback to calibrated linear proxy
            z = self.bias + sum(w * x for w, x in zip(self.weights, norm_feats))
            z_clamped = max(min(z, 20.0), -20.0)
            prob = 1.0 / (1.0 + math.exp(-z_clamped))

        is_approved = prob >= self.confidence_threshold
        
        details = {
            "confidence_score": round(prob, 4),
            "threshold": self.confidence_threshold,
            "approved": is_approved,
            "top_positive_features": [],
            "top_negative_features": []
        }
        
        # Explain top contributors
        contributions = [(self.FEATURE_NAMES[i], self.weights[i] * features[i]) for i in range(len(features))]
        contributions.sort(key=lambda x: x[1], reverse=True)
        details["top_positive_features"] = contributions[:3]
        details["top_negative_features"] = contributions[-3:]
        
        return is_approved, round(prob, 4), details

    def update_model(self, features: List[float], actual_win: bool):
        """
        Online Stochastic Gradient Descent step with L2 Regularization.
        Maintains conservative learning rate (eta=0.015) to avoid overfitting.
        """
        if len(features) != len(self.weights):
            return

        # QUANT FIX: the frozen Res-MLP is authoritative; online SGD on the unused
        # linear proxy would corrupt the persisted linear fallback. Skip when active.
        if self.res_mlp_layers:
            return

        y = 1.0 if actual_win else 0.0
        z = self.bias + sum(w * x for w, x in zip(self.weights, features))
        z_clamped = max(min(z, 20.0), -20.0)
        prob = 1.0 / (1.0 + math.exp(-z_clamped))

        # Gradient of binary cross-entropy: (prob - y)
        error = prob - y

        # Update weights with L2 weight decay: w = w - eta * (error * x + lambda * w)
        for i in range(len(self.weights)):
            grad = (error * features[i]) + (self.l2_lambda * self.weights[i])
            self.weights[i] -= self.learning_rate * grad
            
        # Update bias without regularization
        self.bias -= self.learning_rate * error
        self.total_trained_samples += 1
        self._save_weights()
        print(f"[MLGatekeeper] Online update completed. Sample: {self.total_trained_samples}, Loss Error: {error:.4f}")
