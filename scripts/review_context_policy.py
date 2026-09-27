"""Test sonrası geliştirme deneyi. Çalışan politikanın üstüne yazmaz.

Orijinal test zaten görüldü. Bu deney yalnız validation etiketlerini okuyor,
yani görülmemiş veri üzerinde bir başarı iddiası kuramaz.
"""

import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from fraud_case.context import CONTEXT_RULES, ContextConfig, ContextEngine
from fraud_case.evaluate_context import compare, validation_sections

ROOT = Path(__file__).resolve().parents[1]


def run():
    output = ROOT / "artifacts/context_review"
    output.mkdir(exist_ok=True)
    plan = {
        "scope": "Post-test development experiment. Both validation sections have been observed previously; no independent holdout claim.",
        "policies": ["original", "product_requires_stable_entity", "no_product_discount"],
        "strengths": [0.0, 0.25, 0.5, 1.0],
        "selection": "Early validation only: zero lost TP at frozen threshold AND no loss of TP at equal review budget. Maximize FP removal, then budget TP, then lower strength; prefer no discount on no benefit.",
        "audit": "Later validation development recheck; not used to select candidate.",
        "production_promotion": False,
        "test_labels_read": False,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    features = pd.read_parquet(ROOT / "data/processed/features.parquet", filters=[("split", "=", "validation")]).reset_index(drop=True)
    scores = pd.read_parquet(ROOT / "data/processed/scores.parquet", filters=[("split", "=", "validation")]).reset_index(drop=True)
    source = pd.read_parquet(ROOT / "data/processed/transactions.parquet",
                             columns=["TransactionID", "TransactionDT", "ProductCD", "isFraud"],
                             filters=[("split", "=", "validation")]).reset_index(drop=True)
    assert features.TransactionID.equals(source.TransactionID) and features.TransactionID.equals(scores.TransactionID)
    features["TransactionDT"], features["ProductCD"] = source.TransactionDT, source.ProductCD
    cal, audit, _ = validation_sections(features, 0.5)
    original = json.loads((ROOT / "artifacts/context/evaluation.json").read_text())
    base = ContextConfig(**original["selected_context"])
    threshold = original["threshold"]
    stable = (features.prior_transaction_count.ge(base.minimum_prior_count) & features.usual_amount.eq(1)
              & features.amount_to_prior_mean.between(base.stable_ratio_min, base.stable_ratio_max)
              & scores.entity_raw.le(base.stable_entity_deviation_max) & features.entity_product_is_new.eq(0)).fillna(False)
    candidates, frames = [], {}
    for strength in plan["strengths"]:
        engine = ContextEngine(replace(base, strength=strength), original["product_reference"])
        full = engine.apply(features, scores)
        no_product = engine.apply(features, scores, enabled_rules=set(CONTEXT_RULES)-{"product_typical_amount"})
        for policy in plan["policies"]:
            adjusted = full.adjusted_anomaly_score if policy == "original" else no_product.adjusted_anomaly_score
            if policy == "product_requires_stable_entity":
                adjusted = full.adjusted_anomaly_score.where(stable, no_product.adjusted_anomaly_score)
            frame = pd.DataFrame({"TransactionID": features.TransactionID, "isFraud": source.isFraud,
                                  "raw_anomaly_score": scores.raw_anomaly_score, "adjusted_anomaly_score": adjusted})
            result = compare(frame.loc[cal], threshold)
            effect, budget = result["paired_effect"], result["same_review_budget"]
            eligible = effect["lost_true_positives"] == 0 and budget["adjusted"]["tp"] >= budget["baseline"]["tp"]
            candidate = {"policy": policy, "strength": strength, "eligible": eligible, "calibration": result}
            candidates.append(candidate)
            frames[(policy, strength)] = frame
    eligible = [r for r in candidates if r["eligible"]]
    selected = min(eligible, key=lambda r: (-r["calibration"]["paired_effect"]["removed_false_positives"],
                   -r["calibration"]["same_review_budget"]["adjusted"]["tp"], r["strength"],
                   plan["policies"].index(r["policy"])))
    checked = compare(frames[(selected["policy"], selected["strength"])].loc[audit], threshold)
    summary = {"plan": plan, "threshold": threshold, "candidates": candidates, "selected": selected,
               "later_validation_recheck": checked, "serving_policy_changed": False}
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"selected": {k: selected[k] for k in ("policy", "strength")},
                      "calibration_effect": selected["calibration"]["paired_effect"],
                      "later_validation_effect": checked["paired_effect"],
                      "later_budget": checked["same_review_budget"], "serving_policy_changed": False}, indent=2), flush=True)


if __name__ == "__main__":
    run()
