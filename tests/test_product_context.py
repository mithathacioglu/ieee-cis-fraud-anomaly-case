from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pandas as pd
import pytest

from fraud_case.context import CONTEXT_RULES, ContextConfig, ContextEngine
from fraud_case.product_context import ProductRiskContextEngine


def policy():
    profiles = {}
    for name, fraud in (("high", 200), ("low", 20), ("small", 20)):
        count = 100 if name == "small" else 2000
        rate = (fraud+1000*.05)/(count+1000)
        profiles[name] = {"count": count, "fraud_count": fraud, "smoothed_rate": rate,
                          "log_relative_risk": float(np.clip(np.log(rate/.05), -1, 1)) if count >= 1000 else 0.}
    return {"version": 1, "strength": .1, "global_train_rate": .05, "prior_pseudocount": 1000,
            "minimum_product_support": 1000, "training_end_transaction_dt": 100., "profiles": profiles,
            "base_context_config": asdict(ContextConfig()), "product_reference": {}}


def inputs(product="high", raw=.8):
    features = pd.DataFrame({"TransactionID": [1], "TransactionDT": [200.], "ProductCD": [product],
        "prior_transaction_count": [30.], "amount_to_prior_mean": [1.], "prior_count_1h": [2.],
        "frequent_entity": [0.], "usual_amount": [1.], "entity_hour_probability": [.1],
        "entity_product_is_new": [0.], "entity_email_is_new": [0.], "entity_device_is_new": [0.],
        "calendar_available": [0.], "is_business_hours": [np.nan], "is_weekend": [np.nan]})
    scores = pd.DataFrame({"TransactionID": [1], "raw_anomaly_score": [raw], "column_normalized": [.8],
        "multivariate_normalized": [.8], "entity_raw": [.2],
        **{f"{layer}_contribution": [raw/4] for layer in ("column", "multivariate", "entity", "temporal")}})
    return features, scores


@pytest.mark.parametrize("product,direction", [("high", 1), ("low", -1)])
def test_historical_product_risk_changes_score_in_expected_direction(product, direction):
    f, s = inputs(product)
    result = ProductRiskContextEngine(policy()).apply(f, s)
    assert np.sign(result.product_context_adjustment.item()) == direction
    assert result.product_context_available.item()
    assert s.raw_anomaly_score.item() == .8
    contribution = sum(result[f"context_{layer}_contribution"].item() for layer in ("column", "multivariate", "entity", "temporal"))
    assert contribution + result.product_context_adjustment.item() == pytest.approx(result.adjusted_anomaly_score.item())


@pytest.mark.parametrize("product", ["unseen", "small", None])
def test_unknown_or_unsupported_product_gets_no_risk_adjustment(product):
    f, s = inputs(product)
    result = ProductRiskContextEngine(policy()).apply(f, s)
    assert result.product_context_adjustment.item() == 0
    assert not result.product_context_available.item()


@pytest.mark.parametrize("time", [99., 100.])
def test_profiles_are_unavailable_at_or_before_the_training_cutoff(time):
    f, s = inputs()
    f["TransactionDT"] = time
    result = ProductRiskContextEngine(policy()).apply(f, s)
    assert not result.product_context_available.item()
    assert result.product_context_adjustment.item() == 0


def test_strong_signal_blocks_discount_but_not_upward_risk_adjustment():
    engine = ProductRiskContextEngine(policy())
    f, s = inputs("low")
    s["multivariate_normalized"] = .99
    result = engine.apply(f, s)
    assert result.product_context_reduction_blocked.item()
    assert result.product_context_adjustment.item() == 0
    f["ProductCD"] = "high"
    assert engine.apply(f, s).product_context_adjustment.item() > 0


@pytest.mark.parametrize("product,raw,expected", [("high", .99, 1.), ("low", .01, 0.)])
def test_adjustments_are_bounded_and_clipping_is_reflected_in_explanation(product, raw, expected):
    f, s = inputs(product, raw)
    engine = ProductRiskContextEngine(policy())
    result = engine.apply(f, s)
    explanation = engine.explain(f, s)
    assert result.adjusted_anomaly_score.item() == expected
    assert abs(result.product_context_adjustment.item()) <= .1
    assert explanation["net_adjustment"] == pytest.approx(expected-raw)
    assert explanation["product_risk_context"]["applied_adjustment"] == pytest.approx(expected-raw)


def test_trust_and_calendar_rules_are_still_available():
    f, s = inputs("unseen")
    f["calendar_available"], f["is_business_hours"], f["is_weekend"] = 1., 1., 0.
    external = pd.DataFrame({"TransactionID": [1], "trusted_entity": [True], "trust_observed_at": [150.], "trust_source": ["synthetic"]})
    result = ProductRiskContextEngine(policy()).explain(f, s, external)
    assert {row["rule"] for row in result["behavior_context"]["applied"]} == {"business_hours", "verified_trust"}
    assert result["product_risk_context"]["applied_adjustment"] == 0


def test_corrupt_profiles_and_excess_strength_are_rejected():
    p = policy()
    p["profiles"]["high"]["smoothed_rate"] = .99
    with pytest.raises(ValueError, match="inconsistent"):
        ProductRiskContextEngine(p)
    p = policy()
    p["strength"] = .5
    with pytest.raises(ValueError, match="parameters"):
        ProductRiskContextEngine(p)


def test_zero_product_strength_is_not_the_raw_baseline():
    """Ürün gücünü sıfırlamak raw'a dönmek değildir; iki ayrı ayar var.

    `strength` yalnızca ürün riski ek terimini ölçekliyor. Davranış indirimleri
    `base_context_config` içindeki kendi gücünden geliyor ve `product_typical_amount`
    da ürün profilinde ayrıca devre dışı bırakılıyor. Yani sıfır ürün gücü ne raw
    profiline ne de tam davranış context'ine eşit.
    """
    p = policy()
    p["strength"] = 0.
    f, s = inputs("high")
    # Davranis indirimi tutsun diye entity'yi sik ve alisildik yapiyorum.
    f["frequent_entity"], f["entity_hour_probability"] = 1., .5
    zero_product = ProductRiskContextEngine(p).apply(f, s)
    assert zero_product.product_context_adjustment.item() == 0
    behavior = ContextEngine(ContextConfig(**p["base_context_config"]), p["product_reference"]).apply(f, s)
    raw = ContextEngine(replace(ContextConfig(**p["base_context_config"]), strength=0.),
                        p["product_reference"]).apply(f, s)
    assert raw.adjusted_anomaly_score.item() == s.raw_anomaly_score.item()
    # Sifir urun gucu raw degil: davranis indirimi hala uygulaniyor.
    assert zero_product.adjusted_anomaly_score.item() < raw.adjusted_anomaly_score.item()
    # Tam davranis context'i de degil: karsilastirma product_typical_amount haric yapilmali.
    without_product_rule = ContextEngine(ContextConfig(**p["base_context_config"]), p["product_reference"]).apply(
        f, s, enabled_rules=set(CONTEXT_RULES)-{"product_typical_amount"})
    assert zero_product.adjusted_anomaly_score.item() == pytest.approx(
        without_product_rule.adjusted_anomaly_score.item())
    assert behavior.context_reduction.item() >= without_product_rule.context_reduction.item()


def test_experiment_formula_matches_the_engine_on_both_sides_of_the_cutoff():
    """Deneydeki elle yazılmış formül ile servisin motoru aynı skoru vermeli.

    evaluate_product_context.py inceleme eşiğini train satırlarından hesaplıyor.
    Servis o satırlarda ürün riskini uygulamıyor. Deney uygularsa eşik, servisin
    hiç üretmediği bir skor dağılımından çıkar ve deneyde seçilen güç servise
    taşınmaz. Bu test iki tarafı aynı kapıya bağlıyor.
    """
    p = policy()
    cutoff = p["training_end_transaction_dt"]
    frames, rows = [], []
    for transaction_id, (product, time) in enumerate([("high", 50.), ("high", 200.), ("low", 50.), ("low", 200.)], 1):
        f, s = inputs(product)
        f["TransactionID"], s["TransactionID"], f["TransactionDT"] = transaction_id, transaction_id, time
        frames.append(f)
        rows.append(s)
    f, s = pd.concat(frames, ignore_index=True), pd.concat(rows, ignore_index=True)
    behavior = ContextEngine(ContextConfig(**p["base_context_config"]), p["product_reference"]).apply(
        f, s, enabled_rules=set(CONTEXT_RULES)-{"product_typical_amount"})
    relative = f.ProductCD.astype("string").map({k: v["log_relative_risk"] for k, v in p["profiles"].items()}).fillna(0.)
    relative = relative.mask(behavior.context_guard_active & relative.lt(0), 0.)
    relative = relative.where(f.TransactionDT.gt(cutoff), 0.)
    expected = (behavior.adjusted_anomaly_score+p["strength"]*relative).clip(0, 1)
    served = ProductRiskContextEngine(p).apply(f, s).adjusted_anomaly_score
    pd.testing.assert_series_equal(served, expected, check_names=False)
    train_side = f.TransactionDT.le(cutoff)
    assert served[train_side].equals(behavior.adjusted_anomaly_score[train_side])
    assert not served[~train_side].equals(behavior.adjusted_anomaly_score[~train_side])


def test_policy_is_not_mutated_and_label_columns_are_not_used_at_inference():
    p = policy()
    before = deepcopy(p)
    engine = ProductRiskContextEngine(p)
    f, s = inputs()
    first = engine.apply(f, s)
    f["isFraud"] = 1
    pd.testing.assert_frame_equal(first, engine.apply(f, s))
    assert p == before
