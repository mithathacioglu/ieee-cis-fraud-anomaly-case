"""Her katmanı train'e göre sırala, sonra elde olanların ortalamasını al."""

import numpy as np
import pandas as pd

from fraud_case.anomaly import LAYERS


class ScoreAggregator:
    def __init__(self, weights: dict[str, float] | None = None):
        self.weights = dict(weights) if weights is not None else dict.fromkeys(LAYERS, 0.25)
        if set(self.weights) != set(LAYERS):
            raise ValueError("Provide exactly the four layer weights")
        if any(not np.isfinite(v) or v < 0 for v in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("Weights must be finite and nonnegative, with positive total")
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.reference = None

    def fit(self, train_scores: pd.DataFrame):
        if "split" in train_scores and not train_scores["split"].eq("train").all():
            raise ValueError("Normalization fit accepts training scores only")
        reference = {}
        for layer in LAYERS:
            values = train_scores[f"{layer}_raw"].to_numpy(dtype=float)
            if np.isinf(values).any():
                raise ValueError("Infinite calibration score")
            values = values[np.isfinite(values)]
            if not len(values):
                raise ValueError(f"No training reference for {layer}")
            reference[layer] = np.sort(values)
        self.reference = reference
        return self

    def transform(self, raw_scores: pd.DataFrame) -> pd.DataFrame:
        if self.reference is None:
            raise RuntimeError("Fit the score aggregator before transforming")
        output = raw_scores.copy()
        denominator = np.zeros(len(output))
        numerator = np.zeros(len(output))
        for layer in LAYERS:
            raw = output[f"{layer}_raw"].to_numpy(dtype=float)
            if np.isinf(raw).any():
                raise ValueError("Infinite raw score")
            present = np.isfinite(raw)
            normalized = np.full(len(output), np.nan)
            # Strict rank: sıfıra yığılan skorlar yüksek yüzdelik almasın
            normalized[present] = np.searchsorted(self.reference[layer], raw[present], side="left") / len(self.reference[layer])
            output[f"{layer}_normalized"] = normalized
            denominator += present * self.weights[layer]
            numerator += np.nan_to_num(normalized, nan=0) * self.weights[layer]
        if np.any(denominator == 0):
            raise ValueError("A row has no positively weighted available layer")
        output["available_weight"] = denominator
        output["raw_anomaly_score"] = numerator / denominator
        for layer in LAYERS:
            present = output[f"{layer}_normalized"].notna().to_numpy()
            effective = present * self.weights[layer] / denominator
            output[f"{layer}_effective_weight"] = effective
            output[f"{layer}_contribution"] = output[f"{layer}_normalized"].fillna(0).to_numpy() * effective
        return output
