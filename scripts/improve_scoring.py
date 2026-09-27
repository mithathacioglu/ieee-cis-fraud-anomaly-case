"""Dört katman ağırlığı ve context politikası için sınırlı bir geliştirme denemesi.

Sadece train skorları ve validation etiketleri kullanılıyor. Eski test zaten
görüldü; sonraki validation bağımsız test değil, geliştirme kontrolü.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from fraud_case.aggregation import ScoreAggregator
from fraud_case.anomaly import LAYERS
from fraud_case.context import CONTEXT_RULES, ContextConfig, ContextEngine
from fraud_case.evaluate_context import compare, validation_sections
from fraud_case.evaluation import metrics

ROOT = Path(__file__).resolve().parents[1]


def run():
    folder = ROOT / "artifacts/scoring_development"
    folder.mkdir(parents=True, exist_ok=True)
    plans = {
        "equal": [.25, .25, .25, .25],
        "column_emphasis": [.55, .15, .15, .15],
        "multivariate_emphasis": [.15, .55, .15, .15],
        "entity_emphasis": [.15, .15, .55, .15],
        "temporal_emphasis": [.15, .15, .15, .55],
        "multivariate_entity": [.10, .40, .40, .10],
        "entity_temporal": [.10, .10, .40, .40],
        "multivariate_temporal": [.10, .40, .10, .40],
        "behavior_emphasis": [.10, .30, .30, .30],
    }
    plan = {"weights": {name: dict(zip(LAYERS, values, strict=True)) for name, values in plans.items()},
            "primary_review_fraction": .05, "sensitivity_review_fractions": [.01, .05, .10],
            "weight_selection": "Early validation split into two chronological blocks. Maximize minimum block TP gain against equal weights at identical 5% budgets, then total TP, then AP; ties favor earlier listed candidate.",
            "context_policies": ["original", "product_requires_stable_entity"], "context_strengths": [0., .25, .5, 1.],
            "context_selection": "Early validation: at own train-95%-threshold retain >=98% of raw TP in each block AND no loss of TP at the same 5% budget in either block. Then maximize FP removed, budget TP gain, then lower strength.",
            "later_validation": "Previously observed development recheck only; not used in parameter selection.",
            "test_labels_read": False, "serving_policy_changed": False,
            "scope": "Case steps 5 and 6 only; four original anomaly layers and train normalization references are retained. Validation labels tune weights; the complete tuned system is not purely unsupervised."}
    # Aday listesini validation etiketlerini yüklemeden önce kaydet
    (folder / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    scores = pd.read_parquet(ROOT / "data/processed/scores.parquet", filters=[("split", "in", ["train", "validation"])])
    train = scores.loc[scores["split"].eq("train")].reset_index(drop=True)
    raw = scores.loc[scores["split"].eq("validation")].reset_index(drop=True)
    features = pd.read_parquet(ROOT / "data/processed/features.parquet", filters=[("split", "=", "validation")]).reset_index(drop=True)
    source = pd.read_parquet(ROOT / "data/processed/transactions.parquet", columns=["TransactionID", "TransactionDT", "ProductCD", "isFraud"],
                             filters=[("split", "=", "validation")]).reset_index(drop=True)
    assert features.TransactionID.equals(raw.TransactionID) and features.TransactionID.equals(source.TransactionID)
    features["TransactionDT"], features["ProductCD"] = source.TransactionDT, source.ProductCD
    cal, audit, boundary = validation_sections(features, .5)
    first, second, block_boundary = validation_sections(features.loc[cal], .5)
    block_indices = [first.index[first], second.index[second]]
    original_model = joblib.load(ROOT / "artifacts/scoring/model.joblib")
    old_context = json.loads((ROOT / "artifacts/context/evaluation.json").read_text())
    base = ContextConfig(**old_context["selected_context"])
    reference = old_context["product_reference"]
    def at_budget(scored, index, rate=.05):
        return metrics(source.loc[index, "isFraud"], scored.loc[index], review_count=int(np.ceil(len(index)*rate)),
                       transaction_ids=source.loc[index, "TransactionID"])
    weight_rows, candidates = [], {}
    for name, weights in plan["weights"].items():
        aggregator = ScoreAggregator(weights)
        aggregator.reference = original_model["aggregator"].reference
        transformed = aggregator.transform(raw)
        threshold = float(aggregator.transform(train).raw_anomaly_score.quantile(.95, interpolation="higher"))
        blocks = [at_budget(transformed.raw_anomaly_score, index) for index in block_indices]
        row = {"name": name, "weights": weights, "threshold": threshold, "blocks": blocks,
               "calibration": at_budget(transformed.raw_anomaly_score, features.index[cal])}
        weight_rows.append(row)
        candidates[name] = (aggregator, transformed)
    baseline_blocks = weight_rows[0]["blocks"]
    for row in weight_rows:
        row["minimum_block_tp_gain"] = min(block["tp"]-ref["tp"] for block, ref in zip(row["blocks"], baseline_blocks, strict=True))
    chosen = max(weight_rows, key=lambda r: (r["minimum_block_tp_gain"], r["calibration"]["tp"], r["calibration"]["average_precision"]))
    aggregator, weighted = candidates[chosen["name"]]
    print(f"Selected weights on early validation: {chosen['name']}; TP={chosen['calibration']['tp']}", flush=True)
    stable = (features.prior_transaction_count.ge(base.minimum_prior_count) & features.usual_amount.eq(1)
              & features.amount_to_prior_mean.between(base.stable_ratio_min, base.stable_ratio_max)
              & weighted.entity_raw.le(base.stable_entity_deviation_max) & features.entity_product_is_new.eq(0)).fillna(False)
    context_rows, context_frames = [], {}
    for strength in plan["context_strengths"]:
        engine = ContextEngine(replace(base, strength=strength), reference)
        full = engine.apply(features, weighted)
        without_product = engine.apply(features, weighted, enabled_rules=set(CONTEXT_RULES)-{"product_typical_amount"})
        for policy in plan["context_policies"]:
            adjusted = full.adjusted_anomaly_score if policy == "original" else full.adjusted_anomaly_score.where(stable, without_product.adjusted_anomaly_score)
            frame = pd.DataFrame({"TransactionID": source.TransactionID, "isFraud": source.isFraud,
                                  "raw_anomaly_score": weighted.raw_anomaly_score, "adjusted_anomaly_score": adjusted})
            checks = [compare(frame.loc[index], chosen["threshold"]) for index in block_indices]
            budget_checks = [at_budget(adjusted, index) for index in block_indices]
            eligible = all(check["adjusted"]["tp"] >= .98*check["baseline"]["tp"]
                           and after["tp"] >= before["tp"]
                           for check, after, before in zip(checks, budget_checks, chosen["blocks"], strict=True))
            comparison = compare(frame.loc[cal], chosen["threshold"])
            row = {"policy": policy, "strength": strength, "eligible": eligible, "blocks": checks,
                   "fixed_budget_blocks": budget_checks, "calibration": comparison,
                   "fixed_budget_calibration": at_budget(adjusted, features.index[cal])}
            context_rows.append(row)
            context_frames[(policy, strength)] = frame
    selected_context = max((r for r in context_rows if r["eligible"]), key=lambda r: (
        r["calibration"]["paired_effect"]["removed_false_positives"], r["fixed_budget_calibration"]["tp"], -r["strength"]))
    selected_frame = context_frames[(selected_context["policy"], selected_context["strength"])]
    audit_index = features.index[audit]
    audit_results = {"equal_raw": at_budget(raw.raw_anomaly_score, audit_index),
                     "weighted_raw": at_budget(weighted.raw_anomaly_score, audit_index),
                     "weighted_context": at_budget(selected_frame.adjusted_anomaly_score, audit_index),
                     "context_same_threshold": compare(selected_frame.loc[audit], chosen["threshold"])}
    sensitivity = {str(rate): {"equal_raw": at_budget(raw.raw_anomaly_score, audit_index, rate),
                              "weighted_raw": at_budget(weighted.raw_anomaly_score, audit_index, rate),
                              "weighted_context": at_budget(selected_frame.adjusted_anomaly_score, audit_index, rate)}
                   for rate in plan["sensitivity_review_fractions"]}
    summary = {"plan": plan, "validation_boundary": boundary, "calibration_block_boundary": block_boundary,
               "weight_candidates": weight_rows, "context_candidates": context_rows,
               "selected_weights": chosen, "selected_context": selected_context,
               "later_validation_recheck": audit_results, "sensitivity": sensitivity}
    (folder / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    # Açık bir aday dosyası; çalışan servis bunu kendiliğinden seçmiyor
    joblib.dump({"engine": original_model["engine"], "aggregator": aggregator}, folder / "model.joblib", compress=3)
    (folder / "selected_policy.json").write_text(json.dumps({"weights": chosen["weights"],
        "context_policy": selected_context["policy"], "context_strength": selected_context["strength"],
        "threshold": chosen["threshold"], "status": "development_candidate_not_promoted"}, indent=2), encoding="utf-8")
    for path in (ROOT / "artifacts/scoring/model.joblib", ROOT / "artifacts/context/selected_config.json"):
        print(f"Unchanged serving artifact: {path.name} {hashlib.sha256(path.read_bytes()).hexdigest()}", flush=True)
    print(json.dumps({"weights": chosen["weights"], "context": {k: selected_context[k] for k in ("policy", "strength")},
                      "later_validation": audit_results}, indent=2), flush=True)


if __name__ == "__main__":
    run()
