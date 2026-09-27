"""Context indirimleri.

Katman başına tek kural, üst sınır 0.05 puan ve raw skorun %15'i, guard
devredeyse indirim tamamen kapalı. Amaç bütün skoru bir çarpanla düşürmek
değil, hangi sinyalin yeniden yorumlandığını görebilmek.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fraud_case.anomaly import LAYERS


@dataclass(frozen=True)
class ContextConfig:
    strength: float = 1.0
    max_absolute_reduction: float = 0.05
    max_relative_reduction: float = 0.15
    minimum_prior_count: int = 20
    stable_ratio_min: float = 0.8
    stable_ratio_max: float = 1.25
    stable_entity_deviation_max: float = 1.0
    guard_entity_deviation: float = 3.0
    guard_multivariate_rank: float = 0.98
    guard_hourly_count: int = 10
    frequent_temporal_discount: float = 0.20
    familiar_amount_discount: float = 0.15
    business_hours_discount: float = 0.10
    expected_weekend_discount: float = 0.10
    trusted_entity_discount: float = 0.10
    product_amount_discount: float = 0.10
    product_minimum_support: int = 1000
    product_lower_quantile: float = 0.10
    product_upper_quantile: float = 0.90
    high_amount_rank: float = 0.90

    def __post_init__(self):
        values = asdict(self)
        if not all(np.isfinite(v) for v in values.values()):
            raise ValueError("Context parameters must be finite")
        for name in ("strength", "max_absolute_reduction", "max_relative_reduction", "guard_multivariate_rank",
                     "high_amount_rank", "frequent_temporal_discount", "familiar_amount_discount",
                     "business_hours_discount", "expected_weekend_discount", "trusted_entity_discount", "product_amount_discount"):
            if not 0 <= values[name] <= 1:
                raise ValueError(f"{name} must be in [0,1]")
        if self.minimum_prior_count < 5 or self.guard_hourly_count < 1:
            raise ValueError("Invalid history or velocity threshold")
        if not 0 < self.stable_ratio_min <= 1 <= self.stable_ratio_max:
            raise ValueError("Stable amount interval must contain one")
        if not 0 <= self.stable_entity_deviation_max < self.guard_entity_deviation:
            raise ValueError("Stable deviation must be smaller than the guard")
        if self.product_minimum_support < 1 or not 0 <= self.product_lower_quantile < self.product_upper_quantile <= 1:
            raise ValueError("Invalid product reference configuration")


CONTEXT_FEATURES = [
    "TransactionID", "TransactionDT", "prior_transaction_count", "amount_to_prior_mean",
    "prior_count_1h", "frequent_entity", "usual_amount", "entity_hour_probability",
    "entity_product_is_new", "entity_email_is_new", "entity_device_is_new",
    "calendar_available", "is_business_hours", "is_weekend",
]
EXTERNAL_COLUMNS = ["TransactionID", "trusted_entity", "trust_observed_at", "trust_source",
                    "weekend_activity_expected", "schedule_observed_at", "schedule_source"]
CONTEXT_RULES = {
    "product_typical_amount": "Globally unusual amount lies inside its sufficiently supported product's training reference interval: reduce column contribution.",
    "frequent_familiar_activity": "Frequent entity, familiar amount/relations and observed usual hour phase: reduce temporal contribution.",
    "familiar_high_amount": "Globally unusual amount is ordinary for an established entity: reduce column contribution.",
    "business_hours": "Explicit calendar indicates weekday business hours and entity behavior is stable: reduce temporal contribution.",
    "expected_weekend": "Explicit calendar and a prior external weekend schedule agree with stable activity: reduce temporal contribution.",
    "verified_trust": "Prior external trust evidence and stable behavior: reduce multivariate contribution.",
}


class ContextEngine:
    def __init__(self, config: ContextConfig | None = None, product_reference: dict | None = None):
        self.config = config or ContextConfig()
        self.product_reference = product_reference or {}

    def fit_product_context(self, train: pd.DataFrame):
        if not {"ProductCD", "amount", "split"}.issubset(train.columns) or not train["split"].eq("train").all():
            raise ValueError("Product reference fit requires training partition, ProductCD and amount")
        if not np.isfinite(train.amount).all() or train.amount.lt(0).any():
            raise ValueError("Product reference amounts must be finite and nonnegative")
        reference = {}
        for product, group in train.dropna(subset=["ProductCD"]).groupby("ProductCD", observed=True):
            if len(group) >= self.config.product_minimum_support:
                reference[str(product)] = {"count": len(group),
                    "lower": float(group.amount.quantile(self.config.product_lower_quantile)),
                    "upper": float(group.amount.quantile(self.config.product_upper_quantile))}
        self.product_reference = reference
        return self

    @staticmethod
    def _external(features: pd.DataFrame, external: pd.DataFrame | None) -> tuple[pd.Series, pd.Series]:
        no = pd.Series(False, index=features.index)
        if external is None or external.empty:
            return no, no.copy()
        if "TransactionID" not in external or external.TransactionID.isna().any() or external.TransactionID.duplicated().any():
            raise ValueError("External context requires unique, non-null TransactionID")
        unexpected = set(external.columns) - set(EXTERNAL_COLUMNS)
        if unexpected:
            raise ValueError(f"Unexpected external context fields: {sorted(unexpected)}")
        data = external.set_index("TransactionID").reindex(features.TransactionID).set_axis(features.index)
        result = []
        for flag, timestamp, source in (("trusted_entity", "trust_observed_at", "trust_source"),
                                       ("weekend_activity_expected", "schedule_observed_at", "schedule_source")):
            if flag not in data:
                result.append(no.copy())
                continue
            known = data[flag].dropna()
            if not known.isin([True, False]).all():
                raise ValueError(f"{flag} must be boolean or null")
            enabled = data[flag].eq(True).fillna(False)
            if enabled.any() and not {timestamp, source}.issubset(data.columns):
                raise ValueError(f"{flag} requires a source and observation time")
            if not enabled.any():
                result.append(no.copy())
                continue
            observed_at = pd.to_numeric(data[timestamp], errors="coerce")
            has_source = data[source].astype("string").str.strip().str.len().gt(0).fillna(False)
            has_time = observed_at.notna() & np.isfinite(observed_at) & observed_at.ge(0)
            if (enabled & ~(has_source & has_time)).any():
                raise ValueError(f"{flag} has missing/invalid evidence")
            # İşlemle aynı saniyede veya sonrasında gelen kayıt geçmiş bilgi değil
            result.append(enabled & observed_at.lt(features.TransactionDT))
        return tuple(result)

    def apply(self, features: pd.DataFrame, scores: pd.DataFrame, external: pd.DataFrame | None = None,
              enabled_rules: set[str] | None = None) -> pd.DataFrame:
        missing = set(CONTEXT_FEATURES) - set(features.columns)
        if missing:
            raise ValueError(f"Missing context features: {sorted(missing)}")
        required_scores = {"TransactionID", "raw_anomaly_score", "column_normalized", "multivariate_normalized", "entity_raw"} | {f"{layer}_contribution" for layer in LAYERS}
        if required_scores - set(scores.columns):
            raise ValueError("Missing context scoring inputs")
        if not features.index.equals(scores.index) or not features.TransactionID.equals(scores.TransactionID):
            raise ValueError("Context inputs must have identical row IDs and index")
        if features.TransactionID.duplicated().any() or features.TransactionID.isna().any():
            raise ValueError("Context transaction IDs must be unique and present")
        if not np.isfinite(features.TransactionDT).all():
            raise ValueError("Context transaction times must be finite")
        raw = scores.raw_anomaly_score.to_numpy(dtype=float)
        contribution_columns = [f"{layer}_contribution" for layer in LAYERS]
        contributions = scores[contribution_columns].to_numpy(dtype=float)
        if not np.isfinite(raw).all() or not np.isfinite(contributions).all() or (raw < 0).any() or (raw > 1).any() or (contributions < 0).any():
            raise ValueError("Invalid raw scores or contributions")
        if not np.allclose(contributions.sum(axis=1), raw, rtol=1e-10, atol=1e-12):
            raise ValueError("Layer contributions must sum to the raw anomaly score")
        enabled_rules = set(CONTEXT_RULES) if enabled_rules is None else enabled_rules
        if enabled_rules - set(CONTEXT_RULES):
            raise ValueError("Unknown context rule")
        c = self.config
        trusted, weekend_expected = self._external(features, external)
        novelty = features[["entity_product_is_new", "entity_email_is_new", "entity_device_is_new"]].eq(1).any(axis=1)
        guard = (scores.entity_raw.ge(c.guard_entity_deviation) |
                 scores.multivariate_normalized.ge(c.guard_multivariate_rank) |
                 features.prior_count_1h.ge(c.guard_hourly_count) | novelty).fillna(False)
        stable = (features.prior_transaction_count.ge(c.minimum_prior_count) & features.usual_amount.eq(1) &
                  features.amount_to_prior_mean.between(c.stable_ratio_min, c.stable_ratio_max) &
                  scores.entity_raw.le(c.stable_entity_deviation_max) & features.entity_product_is_new.eq(0) & ~guard).fillna(False)
        frequent = stable & features.frequent_entity.eq(1)
        calendar = features.calendar_available.eq(1)
        rules = [
            ("frequent_familiar_activity", "temporal", frequent & features.entity_hour_probability.ge(1 / 24), c.frequent_temporal_discount),
            ("familiar_high_amount", "column", frequent & scores.column_normalized.ge(c.high_amount_rank), c.familiar_amount_discount),
            ("business_hours", "temporal", stable & calendar & features.is_business_hours.eq(1), c.business_hours_discount),
            ("expected_weekend", "temporal", stable & calendar & features.is_weekend.eq(1) & weekend_expected, c.expected_weekend_discount),
            ("verified_trust", "multivariate", stable & trusted, c.trusted_entity_discount),
        ]
        if self.product_reference:
            if not {"ProductCD", "amount"}.issubset(features.columns):
                raise ValueError("Fitted product context requires ProductCD and amount")
            codes = features.ProductCD.astype("string")
            lower = codes.map({k: v["lower"] for k, v in self.product_reference.items()})
            upper = codes.map({k: v["upper"] for k, v in self.product_reference.items()})
            product_typical = features.amount.ge(lower) & features.amount.le(upper) & ~guard & scores.column_normalized.ge(c.high_amount_rank)
        else:
            product_typical = pd.Series(False, index=features.index)
        rules.insert(0, ("product_typical_amount", "column", product_typical, c.product_amount_discount))
        output = scores.copy()
        discounts = pd.DataFrame(0., index=scores.index, columns=LAYERS)
        reasons = pd.DataFrame("none", index=scores.index, columns=LAYERS)
        for name, layer, mask, rate in rules:
            matched = mask.fillna(False) & (name in enabled_rules)
            proposed = scores[f"{layer}_contribution"] * rate * c.strength
            wins = matched & proposed.gt(discounts[layer])
            discounts.loc[wins, layer] = proposed.loc[wins]
            reasons.loc[wins, layer] = name
            output[f"context_match_{name}"] = matched
        proposed_total = discounts.sum(axis=1).to_numpy()
        limit = np.minimum(c.max_absolute_reduction, raw * c.max_relative_reduction)
        scale = np.divide(np.minimum(proposed_total, limit), proposed_total, out=np.zeros(len(raw)), where=proposed_total > 0)
        discounts = discounts.mul(scale, axis=0)
        for layer in LAYERS:
            output[f"context_{layer}_reduction"] = discounts[layer]
            output[f"context_{layer}_reason"] = reasons[layer].where(discounts[layer].gt(0), "none")
            output[f"context_{layer}_contribution"] = scores[f"{layer}_contribution"] - discounts[layer]
        output["context_reduction"] = discounts.sum(axis=1)
        output["adjusted_anomaly_score"] = (scores.raw_anomaly_score - output.context_reduction).clip(0, 1)
        output["context_guard_active"] = guard
        output["context_calendar_available"] = calendar
        output["context_trust_available"] = trusted
        output["context_weekend_schedule_available"] = weekend_expected
        return output

    def explain(self, features: pd.DataFrame, scores: pd.DataFrame, external: pd.DataFrame | None = None) -> dict:
        if len(features) != 1:
            raise ValueError("Explain expects one transaction")
        row = self.apply(features, scores, external).iloc[0]
        observations = {}
        for name in CONTEXT_FEATURES + ["ProductCD", "amount"]:
            if name not in features:
                continue
            value = features.iloc[0][name]
            observations[name] = None if pd.isna(value) else (float(value) if isinstance(value, (int, float, np.number)) else str(value))
        product = str(features.iloc[0]["ProductCD"]) if "ProductCD" in features else None
        return {
            "transaction_id": int(row.TransactionID), "raw_score": float(row.raw_anomaly_score),
            "adjusted_score": float(row.adjusted_anomaly_score), "reduction": float(row.context_reduction),
            "guard_active": bool(row.context_guard_active),
            "calendar_available": bool(row.context_calendar_available), "trust_available": bool(row.context_trust_available),
            "observed_context": observations, "product_reference": self.product_reference.get(product),
            "config": asdict(self.config),
            "matched_rules": [name for name in CONTEXT_RULES if row[f"context_match_{name}"]],
            "applied": [{"layer": layer, "rule": row[f"context_{layer}_reason"],
                         "reduction": float(row[f"context_{layer}_reduction"])} for layer in LAYERS if row[f"context_{layer}_reduction"] > 0],
            "policy": "Maximum reduction per layer; equal proposals retain first listed rule; global caps scale all reductions proportionally.",
        }
