"""Karar raporlamasının saf parçaları.

Bu dosyadaki testler metrik üretmiyor; raporların sayıdan cümleye geçtiği yerleri
tutuyor. Yanlış çıkarsa rapor yanlış sayı yazar, bu da sessizce geçen bir hata
türü.
"""

import json

import pandas as pd
import pytest
from evaluate_default_policy import equal_volume_threshold, rule_inputs
from evaluate_final import final_report_lines
from evaluate_product_uncertainty import direction, interval_sign, reading

from fraud_case.evaluate_context import decision_boundary


def audit(removed_fp=115, lost_tp=17, added_fp=0, added_tp=0, alerts=2207, adjusted_alerts=2075):
    return {"baseline": {"alerts": alerts}, "adjusted": {"alerts": adjusted_alerts},
            "paired_effect": {"removed_false_positives": removed_fp, "added_false_positives": added_fp,
                              "lost_true_positives": lost_tp, "added_true_positives": added_tp}}


def test_break_even_uses_net_changes_and_separates_cost_models():
    result = decision_boundary(audit(), prevention_rate=.5)
    models = result["models"]
    assert models["false_positive_cost_only"]["break_even"] == pytest.approx(115/17)
    # Kaldirilan inceleme sayisi FP'den fazla: dogru alarmlar da listeden dustu.
    assert result["removed_reviews"] == 132
    assert models["every_review_costs"]["break_even"] == pytest.approx(132/17)
    # Yakalananin yarisi onlenebiliyorsa esik ikiye katlanir.
    assert models["partial_prevention"]["break_even"] == pytest.approx(2*115/17)


def test_break_even_nets_out_newly_added_alerts():
    """Context bazi islemleri esigin uzerine de tasiyabilir; oran net degisimi kullanmali."""
    result = decision_boundary(audit(removed_fp=115, added_fp=15, lost_tp=17, added_tp=2))
    assert result["net_false_positives_removed"] == 100
    assert result["net_true_positives_lost"] == 15
    assert result["models"]["false_positive_cost_only"]["break_even"] == pytest.approx(100/15)


def test_break_even_is_undefined_when_no_fraud_is_lost():
    result = decision_boundary(audit(lost_tp=0))
    assert result["status"] == "no_fraud_lost"
    assert "models" not in result


@pytest.mark.parametrize(("first", "second", "expected"), [
    (4, 19, "aynı"), (-3, -8, "aynı"), (4, -19, "ters"), (-4, 19, "ters"),
    (0, 19, "biri sıfır"), (4, 0, "biri sıfır"), (0, 0, "biri sıfır"),
])
def test_period_direction_does_not_call_a_zero_gap_a_reversal(first, second, expected):
    assert direction(first, second) == expected


def block_states(*bounds):
    """Blok uzunluklari. None = hesaplanamadi, (low, high) = hesaplanan aralik."""
    states = []
    for index, pair in enumerate(bounds):
        if pair is None:
            states.append({"hours": 6 * (index + 1), "computed": False, "sign": None, "ci_low": None, "ci_high": None})
            continue
        low, high = pair
        states.append({"hours": 6 * (index + 1), "computed": True, "ci_low": low, "ci_high": high,
                       "sign": interval_sign(low, high)})
    return states


POSITIVE, NEGATIVE, STRADDLES = (28., 169.), (-169., -28.), (-10., 45.)


@pytest.mark.parametrize(("gap", "bounds", "first", "second", "expected"), [
    # 1) Kazanim: fark pozitif, iki aralik da pozitif yonde, iki yari da pozitif.
    (94, (POSITIVE, POSITIVE), 55, 33, "gain"),
    # 2) Ayni kosullar negatif tarafta bir KAYIP; "kazanim" yazmasi hataydi.
    (-94, (NEGATIVE, NEGATIVE), -55, -33, "loss"),
    # 3) Fark sifir: bu "kanit yok" degil, "olctum ve fark cikmadi".
    (0, (STRADDLES, STRADDLES), 0, 0, "flat"),
    # 4) Yon tutarli ama belirsizlik sifiri kapsiyor.
    (12, (STRADDLES, STRADDLES), 4, 19, "weak"),
    (-12, (STRADDLES, STRADDLES), -4, -19, "weak"),
    # 5) Aralik sifiri disliyor ama nokta tahmininin TERS yonunde: celiski.
    (12, (NEGATIVE, NEGATIVE), 4, 19, "conflict"),
    (-12, (POSITIVE, POSITIVE), -4, -19, "conflict"),
    # 6) Iki blok uzunlugu birbirinin tersini soyluyor: celiski.
    (94, (POSITIVE, NEGATIVE), 55, 33, "conflict"),
    # 7) Fark sifirken aralik sifiri disliyor: celiski.
    (0, (POSITIVE, POSITIVE), 0, 0, "conflict"),
    # 8) Donemler ters yonde, ya da bir yari sifir: kanit yok.
    (12, (POSITIVE, POSITIVE), -4, 19, "none"),
    (12, (POSITIVE, POSITIVE), 0, 19, "none"),
    (-9, (STRADDLES, STRADDLES), 2, -13, "none"),
    # 9) Blok uzunluklarindan biri hesaplanamadi: yarim kanit tam kanit degil.
    (94, (POSITIVE, None), 55, 33, "partial"),
    (-12, (None, NEGATIVE), -6, -6, "partial"),
    # 10) Hicbiri hesaplanamadi: "sifiri kapsiyor" demek yanlis gerekce olur.
    (94, (None, None), 55, 33, "uncomputed"),
    (0, (None, None), 0, 0, "uncomputed"),
])
def test_budget_reading_covers_every_way_numbers_and_intervals_can_disagree(gap, bounds, first, second, expected):
    """Beş soru ayrı ayrı sorulmalı ve hiçbiri diğerinin yerine geçmemeli.

    Hesaplanabildi mi, hepsi hesaplandı mı, sıfırı dışlıyor mu, aralığın yönü
    nokta tahminiyle uyuşuyor mu, iki dönemde yön aynı mı. İlk sürüm yalnızca
    üçüncüsüne bakıyordu; ikinci sürüm yönü ekledi ama aralığın yönünü sormadı,
    beraberliği de "kanıt yok" sayıyordu.
    """
    assert reading(gap, block_states(*bounds), first, second) == expected


@pytest.mark.parametrize(("low", "high", "expected"), [
    (28., 169., 1), (-169., -28., -1), (-10., 45., 0), (0., 45., 0), (-45., 0., 0), (0., 0., 0),
])
def test_interval_sign_treats_a_zero_bound_as_covering_zero(low, high, expected):
    assert interval_sign(low, high) == expected


def test_equal_volume_threshold_reports_the_alerts_a_tie_actually_produces():
    """Esit skorlarda tek bir esik tam kapasite garanti etmez; rapor bunu saklamamali."""
    tied = equal_volume_threshold([.9, .8, .8, .1], 2)
    assert tied["threshold"] == .8
    # Hedef iki alarmdi, esik uc uretiyor. Sayiyi yazdirmadan "hacim esitlendi"
    # demek yanlis olurdu.
    assert tied["threshold_alerts"] == 3 and tied["target_alerts"] == 2


def test_equal_volume_threshold_is_exact_without_ties_and_undefined_out_of_range():
    clean = equal_volume_threshold([.9, .8, .7, .1], 2)
    assert clean["threshold"] == .8 and clean["threshold_alerts"] == 2
    for target in (0, 5):
        assert equal_volume_threshold([.9, .8, .7, .1], target)["threshold"] is None


def test_rule_inputs_rejects_a_conflicting_shared_column():
    features = pd.DataFrame({"TransactionID": [1, 2], "amount": [10., 20.]})
    agreeing = pd.DataFrame({"TransactionID": [1, 2], "raw_anomaly_score": [.1, .2]})
    assert list(rule_inputs(features, agreeing).columns) == ["TransactionID", "amount", "raw_anomaly_score"]
    conflicting = pd.DataFrame({"TransactionID": [1, 3], "raw_anomaly_score": [.1, .2]})
    with pytest.raises(ValueError, match="Conflicting rule input column"):
        rule_inputs(features, conflicting)


def stored_final_result():
    return {"rows": 118108, "threshold_from_train": .85167813,
            "raw": {"alerts": 4681, "tp": 487, "fp": 4194, "precision": .104, "recall": .119,
                    "false_positive_rate": .036, "average_precision": .06},
            "context": {"alerts": 4517, "tp": 459, "fp": 4058, "precision": .101, "recall": .112,
                        "false_positive_rate": .035, "average_precision": .06},
            "rules": {"reviews": 8966, "tp": 581, "fp": 8385, "precision": .064, "recall": .142,
                      "false_positive_rate": .073},
            "same_review_budget": {"raw": {"tp": 487}, "context": {"tp": 475}}}


def test_final_report_states_which_policy_the_rule_row_belongs_to():
    text = final_report_lines(stored_final_result())
    assert "Rules satırı güncel varsayılanın final dönemindeki performansı değildir" in text.replace("**", "")
    assert "default_policy.md" in text
    assert "4,681" in text and "8,966" in text


def test_final_report_needs_no_labels_or_parquet(tmp_path):
    """--report-only yolunun kayitli JSON disinda hicbir girdiye ihtiyaci olmamali."""
    path = tmp_path / "evaluation.json"
    path.write_text(json.dumps(stored_final_result()), encoding="utf-8")
    assert final_report_lines(json.loads(path.read_text(encoding="utf-8"))).startswith("# Final test değerlendirmesi")
