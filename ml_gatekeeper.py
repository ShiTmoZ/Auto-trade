"""
Machine Learning Gatekeeper:
- 15-dimensional Quantitative Feature Vector
- Conservative Learning Rate (0.015) & L2 Regularization to resist noise
- Online Gradient Update & Inference without heavy ML dependencies
"""

import os
import math
import json
from typing import Dict, Any, List, Tuple

class MLGatekeeper:
    FEATURE_NAMES = [
        "vol_ratio",                  # 0: Volume ratio vs 20-SMA
        "fvg_size_pct",              # 1: FVG dollar size / Price
        "has_fvg",                   # 2: Binary FVG flag (0 or 1)
        "candle0_body_ratio",        # 3: Body / Total Candle Range
        "risk_atr_ratio",            # 4: Stop distance / ATR14
        "tokyo_range_pct",           # 5: Tokyo Range / Price
        "hour_utc",                  # 6: Normalized UTC hour (0 to 1)
        "day_of_week",               # 7: Normalized day of week (0 to 1)
        "htf_alignment_strength",    # 8: 1.0 if 1H and 4H agree, else 0.5
        "retest_depth_pct",          # 9: Entry distance from broken level
        "volatility_ratio",          # 10: ATR14 / Baseline ATR
        "dist_from_tokyo_open_pct",  # 11: Distance from session open
        "momentum_3c_pct",           # 12: 3-candle rate of change
        "consecutive_run",           # 13: Number of consecutive trend candles
        "range_expansion_ratio"      # 14: Break candle range vs 10-candle avg range
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
        self._load_or_initialize_weights()

    def _load_or_initialize_weights(self):
        """Loads weights from disk or initializes with robust quantitative defaults."""
        if os.path.exists(self.weights_file):
            try:
                with open(self.weights_file, "r") as f:
                    data = json.load(f)
                    self.weights = data.get("weights", [])
                    self.bias = data.get("bias", 0.0)
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
            with open(self.weights_file, "w") as f:
                json.dump({
                    "weights": self.weights,
                    "bias": self.bias,
                    "samples": self.total_trained_samples,
                    "learning_rate": self.learning_rate,
                    "l2_lambda": self.l2_lambda
                }, f, indent=2)
        except Exception as e:
            print(f"[MLGatekeeper] Could not save weights: {e}")

    def extract_features(
        self,
        signal: Dict[str, Any],
        tokyo_data: Dict[str, Any],
        htf_data: Dict[str, Any],
        recent_candles: List[Dict[str, Any]]
    ) -> List[float]:
        """Extracts the 15-dimensional numerical feature vector from market context."""
        entry = signal["entry_price"]
        current_candle = recent_candles[-1]
        
        # 0. vol_ratio
        f0 = min(signal.get("vol_ratio", 1.0) / 3.0, 2.0)
        # 1. fvg_size_pct
        f1 = (signal.get("fvg_size", 0.0) / entry) * 100.0
        # 2. has_fvg
        f2 = 1.0 if signal.get("has_fvg", False) else 0.0
        # 3. candle0_body_ratio
        c0_body = signal.get("candle0_body", 10.0)
        c0_range = max(signal.get("atr14", 50.0), 10.0)
        f3 = min(c0_body / c0_range, 1.0)
        # 4. risk_atr_ratio
        f4 = min(signal.get("risk_distance", 50.0) / max(signal.get("atr14", 50.0), 1.0), 3.0)
        # 5. tokyo_range_pct
        f5 = (tokyo_data.get("range_usd", 300.0) / entry) * 100.0
        # 6. hour_utc (normalized)
        utc_dt = current_candle["utc_dt"]
        f6 = utc_dt.hour / 24.0
        # 7. day_of_week (normalized)
        f7 = utc_dt.weekday() / 6.0
        # 8. htf_alignment_strength
        f8 = 1.0 if (htf_data.get("trend_1h") == htf_data.get("trend_4h")) else 0.6
        # 9. retest_depth_pct
        f9 = (abs(entry - signal["level_broken"]) / entry) * 100.0
        # 10. volatility_ratio
        f10 = min(current_candle.get("atr14", 50.0) / 100.0, 2.5)
        # 11. dist_from_tokyo_open_pct
        f11 = (abs(entry - tokyo_data.get("open", entry)) / entry) * 100.0
        # 12. momentum_3c_pct
        if len(recent_candles) >= 4:
            p3 = recent_candles[-4]["close"]
            f12 = ((entry - p3) / p3) * 100.0
        else:
            f12 = 0.0
        # 13. consecutive_run
        run_count = 1
        for k in range(len(recent_candles) - 2, max(0, len(recent_candles) - 6), -1):
            if recent_candles[k]["close"] > recent_candles[k]["open"]:
                run_count += 1
            else:
                break
        f13 = min(run_count / 5.0, 1.0)
        # 14. range_expansion_ratio
        f14 = min(current_candle["high"] - current_candle["low"] / max(current_candle.get("atr14", 50.0), 1.0), 3.0)

        return [round(x, 4) for x in [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14]]

    def evaluate_signal(self, features: List[float]) -> Tuple[bool, float, Dict[str, Any]]:
        """
        Calculates win probability using the regularized logistic model.
        Returns: (is_approved, confidence_score, explanation_dict)
        """
        if len(features) != len(self.weights):
            return False, 0.0, {"error": "Feature count mismatch"}

        # Dot product
        z = self.bias + sum(w * x for w, x in zip(self.weights, features))
        # Sigmoid activation with clamp to avoid overflow
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
