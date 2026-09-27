"""Train etiketlerinden öğrenilen, sınırlı ürün riski context'i.

Bu katman gözetimli ve isteğe bağlı. Dört gözetimsiz anomali skorunun yerine
geçmiyor, takvim uydurmuyor, skoru olasılığa çevirmiyor.
"""

import json
from pathlib import Path

import numpy as np

from fraud_case.context import CONTEXT_RULES, ContextConfig, ContextEngine


class ProductRiskContextEngine(ContextEngine):
    def __init__(self, policy):
        if policy.get("version") != 1:
            raise ValueError("Unsupported product-risk policy version")
        self.strength = float(policy["strength"])
        self.global_rate = float(policy["global_train_rate"])
        self.training_end = float(policy["training_end_transaction_dt"])
        self.prior = int(policy["prior_pseudocount"])
        self.minimum_support = int(policy["minimum_product_support"])
        if not (np.isfinite(self.strength) and 0 <= self.strength <= .10 and 0 < self.global_rate < 1
                and np.isfinite(self.training_end) and self.training_end >= 0 and self.prior > 0 and self.minimum_support > 0):
            raise ValueError("Invalid product-risk policy parameters")
        self.profiles = {}
        for name, profile in policy["profiles"].items():
            count, fraud = profile["count"], profile["fraud_count"]
            if not isinstance(count, int) or not isinstance(fraud, int) or not 0 <= fraud <= count or count <= 0:
                raise ValueError("Invalid product profile counts")
            rate = (fraud+self.prior*self.global_rate)/(count+self.prior)
            relative = float(np.clip(np.log(rate/self.global_rate), -1, 1)) if count >= self.minimum_support else 0.
            if not np.isclose(rate, profile["smoothed_rate"]) or not np.isclose(relative, profile["log_relative_risk"]):
                raise ValueError("Product profile is inconsistent with its training counts")
            self.profiles[str(name)] = {**profile}
        if not self.profiles:
            raise ValueError("Product-risk policy requires training profiles")
        super().__init__(ContextConfig(**policy["base_context_config"]), policy["product_reference"])

    @classmethod
    def from_json(cls, path):
        return cls(json.loads(Path(path).read_text(encoding="utf-8")))

    def apply(self, features, scores, external=None, enabled_rules=None):
        active = set(CONTEXT_RULES) | {"product_risk"} if enabled_rules is None else set(enabled_rules)
        if active - set(CONTEXT_RULES) - {"product_risk"}:
            raise ValueError("Unknown context rule")
        # Eski ürün-tutar indirimi bununla toplanmıyor, yerine geçiyor
        base = super().apply(features, scores, external, active-set({"product_typical_amount", "product_risk"}))
        if "ProductCD" not in features:
            raise ValueError("Product-risk context requires ProductCD")
        product = features.ProductCD.astype("string")
        count = product.map({key: row["count"] for key, row in self.profiles.items()})
        relative = product.map({key: row["log_relative_risk"] for key, row in self.profiles.items()}).fillna(0.).astype(float)
        available = (count.ge(self.minimum_support) & features.TransactionDT.gt(self.training_end)).fillna(False)
        proposed = self.strength*relative.where(available & ("product_risk" in active), 0.)
        blocked = proposed.lt(0) & base.context_guard_active
        proposed = proposed.mask(blocked, 0.)
        output = base.copy()
        adjusted = (base.adjusted_anomaly_score+proposed).clip(0, 1)
        output["product_context_adjustment"] = adjusted-base.adjusted_anomaly_score
        output["product_context_available"] = available
        output["product_context_reduction_blocked"] = blocked
        output["product_context_log_relative_risk"] = relative.where(available, 0.)
        output["behavior_adjusted_anomaly_score"] = base.adjusted_anomaly_score
        output["adjusted_anomaly_score"] = adjusted
        output["context_net_adjustment"] = adjusted-scores.raw_anomaly_score
        # Katman katkıları davranış düzeltmesini açıklıyor.
        # Ürün riski ayrı bir ek terim; katmana uydurma attribution yok.
        return output

    def explain(self, features, scores, external=None):
        if len(features) != 1:
            raise ValueError("Explain expects one transaction")
        # Ayrı base örneği, ProductRisk.apply'ın iki kez çalışmasını önlüyor
        base_engine = ContextEngine(self.config, self.product_reference)
        disabled_product = ContextConfig(**{**self.config.__dict__, "product_amount_discount": 0.})
        base_engine.config = disabled_product
        behavior = base_engine.explain(features, scores, external)
        behavior["matched_rules"] = [r for r in behavior["matched_rules"] if r != "product_typical_amount"]
        row = self.apply(features, scores, external).iloc[0]
        profile = self.profiles.get(str(features.iloc[0].ProductCD))
        return {"transaction_id": int(row.TransactionID), "raw_score": float(row.raw_anomaly_score),
                "adjusted_score": float(row.adjusted_anomaly_score), "net_adjustment": float(row.context_net_adjustment),
                "behavior_context": behavior,
                "product_risk_context": {"available": bool(row.product_context_available),
                    "profile": dict(profile) if profile is not None else None,
                    "global_training_rate": self.global_rate, "training_end_transaction_dt": self.training_end,
                    "strength": self.strength, "applied_adjustment": float(row.product_context_adjustment),
                    "negative_adjustment_blocked": bool(row.product_context_reduction_blocked),
                    "interpretation": "Smoothed historical training-label product risk; final score is not a fraud probability."},
                "contribution_identity": "Sum of behavioral context layer contributions + product_context_adjustment = adjusted_score."}
