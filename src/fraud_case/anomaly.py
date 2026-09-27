"""Dört skor katmanı. Büyük değer daha sıra dışı demek.

Entity ve temporal, min_history altında sıfır değil null döner; geçmişi
olmayan bir işlem bilinmiyordur, güvenli değil.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer

LAYERS = ("column", "multivariate", "entity", "temporal")
MULTIVARIATE_FEATURES = [
    "log_amount", "hour_sin", "hour_cos", "has_identity", "entity_known",
    "prior_transaction_count", "prior_mean_amount", "prior_std_amount",
    "amount_to_prior_mean", "prior_count_1h", "prior_count_24h", "seconds_since_previous",
    "prior_distinct_products", "prior_distinct_email_domains", "prior_distinct_devices",
    "product_prior_frequency", "email_prior_frequency", "device_prior_frequency",
    "product_card_prior_frequency", "email_pair_prior_frequency", "email_domains_match",
]
LOG_FEATURES = ["prior_transaction_count", "prior_mean_amount", "prior_std_amount", "amount_to_prior_mean",
                "prior_count_1h", "prior_count_24h", "seconds_since_previous",
                "prior_distinct_products", "prior_distinct_email_domains", "prior_distinct_devices"]
REQUIRED_FEATURES = sorted(set(MULTIVARIATE_FEATURES) | {
    "amount", "entity_product_is_new", "entity_email_is_new", "entity_device_is_new", "entity_hour_probability",
})


@dataclass(frozen=True)
class RobustReference:
    center: float
    scale: float
    method: str

    @classmethod
    def fit(cls, values: pd.Series):
        finite = values[np.isfinite(values)]
        if not len(finite):
            raise ValueError("No finite training observations for robust reference")
        median = float(finite.median())
        q10, q25, q75, q90 = finite.quantile([0.10, 0.25, 0.75, 0.90])
        scale = float((q75 - q25) / 1.349)
        method = "IQR / 1.349"
        if scale < 1e-8:
            scale = float((q90 - q10) / 2.563)
            method = "(p90-p10) / 2.563"
        if scale < 1e-8:
            scale, method = 1.0, "unit scale for concentrated/discrete distribution"
        return cls(median, scale, method)

    def deviation(self, values: pd.Series, *, upper_only: bool = False):
        standardized = (values - self.center) / self.scale
        return standardized.clip(lower=0) if upper_only else standardized.abs()


class AnomalyEngine:
    def __init__(self, min_history: int = 5, n_estimators: int = 128, max_samples: int = 1024, random_state: int = 42):
        if min_history < 2:
            raise ValueError("min_history must be at least two")
        self.min_history = min_history
        self.imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
        self.forest = IsolationForest(n_estimators=n_estimators, max_samples=max_samples,
                                      contamination="auto", random_state=random_state, n_jobs=1)
        self.references = None

    @staticmethod
    def validate(features: pd.DataFrame) -> None:
        missing = set(REQUIRED_FEATURES) - set(features.columns)
        if missing:
            raise ValueError(f"Missing scoring features: {sorted(missing)}")
        if features.empty:
            raise ValueError("Empty scoring batch")
        values = features[REQUIRED_FEATURES].to_numpy(dtype=float)
        if np.isinf(values).any():
            raise ValueError("Infinite scoring feature")
        if not features[["amount", "log_amount"]].notna().all().all():
            raise ValueError("Current amount features are required")
        if (features[["amount", "log_amount", *LOG_FEATURES]] < 0).any().any():
            raise ValueError("Negative amount or history quantity")
        probability = features.entity_hour_probability.dropna()
        if not probability.between(0, 1, inclusive="right").all():
            raise ValueError("entity_hour_probability must be in (0,1]")

    @staticmethod
    def matrix(features: pd.DataFrame) -> pd.DataFrame:
        matrix = features[MULTIVARIATE_FEATURES].copy()
        matrix[LOG_FEATURES] = np.log1p(matrix[LOG_FEATURES])
        return matrix

    def fit(self, train: pd.DataFrame):
        self.validate(train)
        if "split" in train and not train["split"].eq("train").all():
            raise ValueError("Fit accepts the training partition only")
        eligible = train.prior_transaction_count.ge(self.min_history)
        if not eligible.any():
            raise ValueError("Training data has no entities with sufficient history")
        history = train.loc[eligible]
        self.references = {
            "amount": RobustReference.fit(train.log_amount),
            "velocity_1h": RobustReference.fit(np.log1p(history.prior_count_1h)),
            "velocity_24h": RobustReference.fit(np.log1p(history.prior_count_24h)),
            "hour_rarity": RobustReference.fit(-np.log(history.entity_hour_probability)),
        }
        matrix = self.imputer.fit_transform(self.matrix(train))
        self.forest.fit(matrix)
        return self

    def score(self, features: pd.DataFrame) -> pd.DataFrame:
        if self.references is None:
            raise RuntimeError("Fit the anomaly engine before scoring")
        self.validate(features)
        output = pd.DataFrame(index=features.index)
        output["column_raw"] = self.references["amount"].deviation(features.log_amount)
        matrix = self.imputer.transform(self.matrix(features))
        # sklearn'de anormal olan daha düşük çıkıyor, yönü ters çevir
        output["multivariate_raw"] = -self.forest.score_samples(matrix)
        eligible = features.prior_transaction_count.ge(self.min_history)
        scale = pd.concat([features.prior_std_amount, features.prior_mean_amount * 0.1,
                           pd.Series(1.0, index=features.index)], axis=1).max(axis=1)
        output["entity_amount_deviation"] = ((features.amount - features.prior_mean_amount).abs() / scale).where(eligible)
        novelties = features[["entity_product_is_new", "entity_email_is_new", "entity_device_is_new"]]
        output["entity_relation_novelty"] = (2.0 * novelties.max(axis=1)).where(eligible)
        output["entity_raw"] = output[["entity_amount_deviation", "entity_relation_novelty"]].max(axis=1)
        for name, field in (("velocity_1h", "prior_count_1h"), ("velocity_24h", "prior_count_24h")):
            output[f"temporal_{name}"] = self.references[name].deviation(np.log1p(features[field]), upper_only=True).where(eligible)
        output["temporal_hour_rarity"] = self.references["hour_rarity"].deviation(-np.log(features.entity_hour_probability), upper_only=True).where(eligible)
        temporal = ["temporal_velocity_1h", "temporal_velocity_24h", "temporal_hour_rarity"]
        output["temporal_raw"] = output[temporal].max(axis=1)
        for layer in LAYERS:
            output[f"{layer}_available"] = output[f"{layer}_raw"].notna()
        return output

    def explain(self, row: pd.DataFrame) -> dict:
        if len(row) != 1:
            raise ValueError("Explain accepts exactly one feature row")
        scored = self.score(row).iloc[0]
        fields = ["amount", "log_amount", "prior_transaction_count", "prior_mean_amount", "prior_std_amount",
                  "prior_count_1h", "prior_count_24h", "entity_hour_probability",
                  "entity_product_is_new", "entity_email_is_new", "entity_device_is_new"]
        evidence = {name: float(row.iloc[0][name]) if pd.notna(row.iloc[0][name]) else None for name in fields}
        scores = {layer: float(scored[f"{layer}_raw"]) if scored[f"{layer}_available"] else None for layer in LAYERS}
        components = {name: float(value) if pd.notna(value) else None for name, value in scored.items()
                      if name.startswith(("entity_", "temporal_")) and not name.endswith(("_raw", "_available"))}
        return {
            "raw_scores": scores, "observed_features": evidence, "components": components,
            "column_reference": self.references["amount"].__dict__,
            "temporal_references": {k: v.__dict__ for k, v in self.references.items() if k != "amount"},
            "entity_scale": "max(prior_std, 0.1 * prior_mean, 1 original amount unit)",
            "minimum_history": self.min_history,
            "multivariate_method": "Negative IsolationForest score_samples on the explicit feature allowlist; no per-feature attribution is claimed.",
            "unavailable_reason": "Entity/temporal layers require sufficient observed entity history.",
        }
