"""Raporların tamamını, sayıları kendim kurduğum özetlerle sınar.

Bu dosyanın ayrı durmasının nedeni bir örüntü. Bu projede sayıdan cümle üreten
kodda üç kez aynı hata çıktı: koşulsuz sonuç cümlesi, yönü sorulmayan karar,
sessizce atılan eksik kanıt. Üçünde de yardımcı fonksiyonun testi vardı,
belgenin testi yoktu. Parçalar tek tek doğruyken birleşen metin yanlış
olabiliyor, çünkü yanlış olan şey cümlenin kendisi.

Buradaki testler rapor metnini baştan sona üretiyor ve sayılarla çelişen bir
cümle varsa düşüyor. Girdiler bilinçli olarak teslimdeki veriden farklı: zarar
senaryosu, hesaplanamayan bootstrap, iki ölçütün de kazandığı durum ve
değiştirilmiş bütçe yapılandırması.
"""

import evaluate_layer_contribution as layers
import evaluate_product_uncertainty as uncertainty
import pytest
import report_temporal_validation as temporal
import reselect_context_strength as criterion

RATES = ("0.01", "0.05", "0.1")
HOURS = (6, 24)
COMPARISONS = {"behavior_minus_raw": ["raw", "behavior_context"],
               "product_minus_raw": ["raw", "product_context"],
               "product_minus_behavior": ["behavior_context", "product_context"]}


def uncertainty_summary(gaps, *, excludes=True, computed=(True, True), halves=None, bounds=None):
    """gaps: butce -> urun/ham TP farki. computed: her blok uzunlugu icin durum.

    bounds: butce -> (low, high). Verilmezse nokta tahmininin etrafinda, onunla
    ayni yonde bir aralik kurulur.
    """
    halves = halves or {rate: (gap // 2, gap - gap // 2) for rate, gap in gaps.items()}
    observed = {rate: {"reviews": 1000, "tp": {"raw": 100, "behavior_context": 100, "product_context": 100 + gap},
                       "gaps": {name: gap for name in COMPARISONS}} for rate, gap in gaps.items()}
    blocks = {}
    for hours, ok in zip(HOURS, computed, strict=True):
        for rate, gap in gaps.items():
            low, high = (bounds or {}).get(rate, (gap - 1., gap + 1.) if excludes else (gap - 50., gap + 50.))
            comparisons = {name: {"observed_tp_gap": gap, "ci_low": low, "ci_high": high,
                                  "excludes_zero": low > 0 or high < 0} for name in COMPARISONS}
            blocks[f"{hours}h_at_{rate}"] = ({"status": "ok", "hours": hours, "blocks": 20, "comparisons": comparisons}
                                             if ok else
                                             {"status": "insufficient_blocks", "hours": hours, "blocks": 3,
                                              "comparisons": {}})
    return {
        "scope": "test", "segment_rows": 59054, "segment_positives": 2061,
        "validation_audit_boundary_seconds": 1., "half_boundary_seconds": 2.,
        "protocol": {"block_hours": list(HOURS), "bootstrap_resamples": 1000, "min_blocks": 8,
                     "review_rates": [float(r) for r in gaps], "primary_review_rate": .05, "seed": 42},
        "comparisons": COMPARISONS, "observed": observed, "block_bootstrap": blocks,
        "chronological_halves": {half: {rate: {"reviews": 500, "tp": {}, "gaps": {name: halves[rate][index]
                                                                                 for name in COMPARISONS}}
                                       for rate in gaps}
                                for index, half in enumerate(("first_half", "second_half"))},
        "primary_gap_sign_consistent": {name: True for name in COMPARISONS},
        "limitations": ["test"], "test_labels_used": False, "input_sha256": {},
    }


def test_opening_sentence_follows_the_observed_signs():
    """Ürün modeli zarar verirken giriş "daha fazla fraud yakalıyor" diyemez."""
    text = uncertainty.report_lines(uncertainty_summary({"0.01": -9, "0.05": -12, "0.1": -4}))
    opening = text.split("\n\n")[1]
    assert "daha az" in opening and "daha fazla" not in opening
    assert "(-12)" in opening and "%5" in opening


def test_opening_sentence_names_both_directions_when_they_disagree():
    text = uncertainty.report_lines(uncertainty_summary({"0.01": -9, "0.05": 12, "0.1": 94}))
    opening = text.split("\n\n")[1]
    assert "daha fazla" in opening and "daha az" in opening
    assert "%1 (-9)" in opening


def test_partial_bootstrap_evidence_is_not_read_as_a_verdict():
    """Bir blok uzunluğu hesaplanamadıysa diğeri tek başına karar vermemeli."""
    text = uncertainty.report_lines(uncertainty_summary({"0.05": 94}, computed=(True, False)))
    assert uncertainty.READINGS["partial"] in text
    assert "1/2" in text
    assert "kazanım var" not in text


def test_uncomputed_interval_is_not_reported_as_covering_zero():
    """Hiç aralık hesaplanmadıysa "belirsizlik sıfırı kapsıyor" yanlış gerekçedir."""
    text = uncertainty.report_lines(uncertainty_summary({"0.05": 94}, computed=(False, False)))
    assert uncertainty.READINGS["uncomputed"] in text
    assert "sıfıra yakın olduğu anlamına gelmez" in text
    assert "0/2" in text and "hesaplanamadı" in text
    assert "sıfırı kapsıyor" not in text and "kazanım var" not in text


def test_loss_is_reported_as_loss_end_to_end():
    text = uncertainty.report_lines(uncertainty_summary({"0.05": -12}))
    assert uncertainty.READINGS["loss"] in text
    assert "zarar yönünde" in text and "kazanım var" not in text
    assert "evet, negatif yönde" in text


def test_interval_pointing_the_other_way_is_reported_as_a_conflict():
    """Aralık sıfırı dışlasa bile nokta tahmininin tersi yöndeyse kazanım yazılamaz."""
    text = uncertainty.report_lines(uncertainty_summary({"0.05": 12}, bounds={"0.05": (-30., -2.)}))
    assert uncertainty.READINGS["conflict"] in text
    assert "kazanım var" not in text and "zarar var" not in text
    assert "uyuşmuyor" in text


def test_a_zero_difference_is_reported_as_measured_not_as_unknown():
    """Beraberlik "kanıt yok" değil; ölçüldü ve fark çıkmadı."""
    text = uncertainty.report_lines(uncertainty_summary({"0.05": 0}, bounds={"0.05": (-10., 10.)}))
    assert uncertainty.READINGS["flat"] in text
    assert "ölçemedim demek değil" in text
    assert uncertainty.READINGS["none"] not in text


def criterion_summary(shipped_gain, best_gain, *, shipped=.5, selected=.0):
    candidates = [{"strength": .0, "at_frozen_threshold": {}, "at_equal_capacity": {"average_precision": .06, "tp": 100},
                   "recall_drop": 0., "false_positives_removed": 0, "detected_fraud_lost": 0,
                   "relative_detected_fraud_loss": 0., "capacity_tp_gain": 0}]
    for strength, gain in ((shipped, shipped_gain), (selected, best_gain)):
        if strength == 0.:
            candidates[0]["capacity_tp_gain"] = gain
            continue
        candidates.append({"strength": strength, "at_frozen_threshold": {},
                           "at_equal_capacity": {"average_precision": .06, "tp": 100 + gain},
                           "recall_drop": .009, "false_positives_removed": 97, "detected_fraud_lost": 23,
                           "relative_detected_fraud_loss": .0833, "capacity_tp_gain": gain})
    return {
        "purpose": "test", "not_a_preregistration": "test", "calibration_rows": 59054,
        "frozen_threshold": .85167813, "equal_capacity_reviews": 2436,
        "baseline_at_threshold": {}, "baseline_at_equal_capacity": {},
        "candidates": candidates,
        "original_criterion": {"rule": "test", "selected_strength": shipped, "shipped_strength": shipped, "flaw": "test"},
        "capacity_criterion": {"rule": "test", "selected_strength": selected, "capacity_tp_gain": best_gain},
        "criteria_agree": shipped == selected, "serving_default": "raw",
        "frozen_outputs_written": False, "test_labels_used": False, "input_sha256": {},
    }


SETUP = {"candidate_strengths": [0., .25, .5, 1.], "maximum_calibration_recall_drop": .01,
         "validation_calibration_fraction": .5}


def test_criterion_report_does_not_deny_a_gain_that_actually_exists():
    """İki ölçütün farklı güç seçmesi, verideki kazanım yok demek değildir."""
    text = criterion.report_lines(criterion_summary(1, 2, shipped=.5, selected=.25), SETUP)
    # Yanlis olan iddia bu: kazanc varken "kazanim degil" demek.
    assert "verideki bir kazanım değil" not in text
    assert "ölçütün eksik kurulmasıydı" not in text
    assert "diyemem" in text and "+1" in text and "+2" in text


def test_criterion_report_points_at_another_strength_when_the_shipped_one_loses():
    text = criterion.report_lines(criterion_summary(-5, 2, shipped=.5, selected=.25), SETUP)
    assert "başka bir gücü işaret ediyor" in text
    assert "verideki bir kazanım değil" not in text


def test_criterion_report_blames_the_criterion_only_when_no_candidate_gains():
    """Teslimdeki veri şekli: her güç eşit kapasitede kaybediyor."""
    text = criterion.report_lines(criterion_summary(-19, 0, shipped=.5, selected=.0), SETUP)
    assert "ölçütün eksik kurulmasıydı" in text and "-19" in text


def test_criterion_report_states_agreement_when_both_criteria_pick_the_same():
    text = criterion.report_lines(criterion_summary(0, 0, shipped=.0, selected=.0), SETUP)
    assert "İki ölçüt de 0.0 gücünü seçiyor" in text


def temporal_result(primary_rate, share, rates=(.01, .05, .1)):
    models = {}
    for name in temporal.model_names({"hybrid_supervised_share": share}):
        models[name] = {"average_precision": .1, "roc_auc": .7,
                        "budgets": {str(r): {"tp": 10, "reviews": 100} for r in rates}}
    fold = {"label_delay_days": 0, "fold": 1, "rows": 1000, "positives": 40, "prevalence": .04,
            "train_rows": 5000, "train_end": 10., "evaluation_start": 11., "evaluation_end": 20.,
            "feature_columns": {"full": ["a"], "without_time": ["a"], "matched": ["a"]}, "models": models,
            "block_bootstrap": [{"hours": h, "blocks": 20, "status": "ok",
                                 "comparisons": {name: {"observed_tp_gap": 1, "ci_low": .5, "ci_high": 1.5}
                                                 for name in ("matched_gb_minus_raw", "full_gb_minus_raw",
                                                              "without_time_gb_minus_full", "matched_lr_minus_raw")}}
                                for h in HOURS],
            "overlap_raw_matched_gb": {"reviews_each": 50, "shared_reviews": 30, "raw_only_tp": 2,
                                       "supervised_only_tp": 3, "union_reviews": 70, "union_tp": 8}}
    return {"manifest": {"protocol": {"folds": 3, "label_delay_days": [0], "block_hours": list(HOURS),
                                      "review_rates": list(rates), "primary_review_rate": primary_rate,
                                      "hybrid_supervised_share": share, "bootstrap_resamples": 1000,
                                      "min_blocks": 8, "seed": 42}},
            "folds": [fold]}


@pytest.mark.parametrize(("primary", "share"), [(.05, .5), (.10, .3), (.01, .25)])
def test_temporal_report_labels_follow_the_configuration(primary, share):
    """Bütçe ve kota yapılandırılabilir; başlıklar da onları izlemek zorunda."""
    text = temporal.report_lines(temporal_result(primary, share))
    expected = f"%{primary*100:g}"
    assert f"## {expected} inceleme bütçesi" in text
    assert f"Aralıklar {expected} bütçedeki TP farkına aittir" in text
    assert f"Her iki liste ayrı ayrı {expected} bütçeye sahiptir" in text
    assert f"Hibrit / %{share*100:g} kota" in text
    for other in (.05, .10, .01):
        if other != primary:
            assert f"## %{other*100:g} inceleme bütçesi" not in text


def test_layer_interpretation_does_not_call_a_consistent_negative_gap_harmless():
    """Aralık sıfırı kapsıyor olsa da iki dönemde tutarlı kayıp "kayıp vermiyor" değildir."""
    readings = {"without_multivariate_minus_raw": "weak", "without_entity_minus_raw": "none",
                "without_column_minus_raw": "loss", "only_temporal_minus_raw": "loss"}
    summary = {"observed": {"0.05": {"gaps": {"without_multivariate_minus_raw": -29,
                                             "without_entity_minus_raw": 2,
                                             "without_column_minus_raw": -60,
                                             "only_temporal_minus_raw": -134}}}}
    text = " ".join(layers.interpretation(summary, readings, "0.05"))
    assert "işaret kayıp yönünde" in text and "multivariate" in text.split("işaret kayıp yönünde")[1]
    assert "kayıp işareti bile görülmeyen" in text and "entity" in text.split("kayıp işareti bile görülmeyen")[1]
    assert "ölçülebilir kayıp veren" in text
    # En buyuk gozlenen sayi, kaniti en guclu sayi degil; bunu ayrica yaziyor.
    assert "En büyük gözlenen etki temporal" in text


def test_layer_interpretation_flags_a_positive_weak_gap_as_no_loss_signal():
    readings = {"without_column_minus_raw": "weak"}
    summary = {"observed": {"0.05": {"gaps": {"without_column_minus_raw": 38}}}}
    text = " ".join(layers.interpretation(summary, readings, "0.05"))
    assert "kayıp işareti bile görülmeyen" in text
    assert "işaret kayıp yönünde" not in text


def test_temporal_budget_columns_follow_the_configured_rates():
    text = temporal.report_lines(temporal_result(.05, .5, rates=(.02, .05)))
    assert "| TP %2 | TP %5 |" in text
    assert "TP %10" not in text
