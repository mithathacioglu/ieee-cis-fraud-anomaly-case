"""İndirim gücünü, sonradan doğru bulduğum ölçütle yeniden seçer.

Neden ayrı bir betik: context deneyine başlarken başarı ölçütünü "sabit eşikte
false positive azalt, recall en fazla 1 yüzde puan düşsün" diye kurmuştum.
Hedef tarafi case'in istedigi seydi; azaltilacak olan false positive. Eksik olan
kisit tarafi: kaybi mutlak recall puaniyla sinirlamak, baseline recall %8
civarindayken yakalanan fraud'un yaklasik onda birini kaybetmeye izin veriyor.

Birincil olcut verilen inceleme kapasitesinde yakalanan fraud olmaliydi:
kapasite sabitken daha cok fraud yakalayan siralama daha iyidir, daha az alarm
uretmek tek basina kazanc degildir. O karsilastirmayi yaptim ama secimden
sonra; once yapmam gerekirdi.

Asagidaki yeniden secimde AYRI bir goreli kayip siniri uygulamiyorum. Kapasite
sabitken TP farki kaybi da kazanci da ayni sayida olctugu icin ikinci bir
tolerans gereksiz olurdu. Goreli kaybi aday basina raporluyorum ama eleme
filtresi olarak kullanmiyorum; secim sirasi kapasitedeki TP farki, sonra AP,
sonra kucuk guc.

Bu betik ölçütü sonradan pre-registered gibi göstermiyor. İlk ölçütle aynı
veride hangi gücün seçildiğini ve doğru ölçütle hangisinin seçileceğini yan yana
koyuyor. Kullandığı veri ilk seçimin kullandığı calibration yarısının aynısı, o
yüzden bu yeni ölçütün bağımsız doğrulaması da değil. Ölçülen şey ölçütün
kendisinin kararı nasıl değiştirdiği.

Dondurulmuş çıktıların hiçbirine yazmıyor: selected_config.json,
context_scores.parquet ve final artifact'ları olduğu gibi kalıyor.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pandas as pd

from fraud_case.context import ContextConfig, ContextEngine
from fraud_case.evaluate_context import validation_sections
from fraud_case.evaluation import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = ContextConfig(**json.loads((ROOT / "config/context.json").read_text(encoding="utf-8")))
    setup = json.loads((ROOT / "config/context_evaluation.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / "artifacts/context/evaluation.json").read_text(encoding="utf-8"))
    features = pd.read_parquet(ROOT / "data/processed/features.parquet")
    scores = pd.read_parquet(ROOT / "data/processed/scores.parquet")
    times = pd.read_parquet(ROOT / "data/processed/transactions.parquet", columns=["TransactionID", "TransactionDT", "ProductCD"])
    if not features.TransactionID.equals(times.TransactionID):
        raise ValueError("Feature/time alignment mismatch")
    features["TransactionDT"], features["ProductCD"] = times.TransactionDT, times.ProductCD
    reference = ContextEngine(base).fit_product_context(features.loc[features["split"].eq("train")]).product_reference
    labels = pd.read_parquet(ROOT / "data/processed/transactions.parquet", columns=["TransactionID", "isFraud"],
                             filters=[("split", "=", "validation")])
    validation = features.loc[features["split"].eq("validation")]
    cal_mask, _, _ = validation_sections(validation, setup["validation_calibration_fraction"])
    cal_index = validation.index[cal_mask]
    y = labels.set_index("TransactionID").reindex(validation.TransactionID).set_axis(validation.index)["isFraud"]
    if y.isna().any():
        raise ValueError("Validation label alignment mismatch")
    threshold = frozen["threshold"]
    y_cal = y.loc[cal_index]
    raw_cal = scores.loc[cal_index].raw_anomaly_score
    ids_cal = features.loc[cal_index].TransactionID
    baseline_threshold = metrics(y_cal, raw_cal, threshold=threshold)
    # Kapasite, ham skorun ayni esikte urettigi alarm sayisi. Boylece iki
    # yontem tam olarak ayni sayida islemi incelemeye gonderiyor.
    capacity = baseline_threshold["alerts"]
    baseline_capacity = metrics(y_cal, raw_cal, review_count=capacity, transaction_ids=ids_cal)
    rows = []
    for strength in setup["candidate_strengths"]:
        engine = ContextEngine(replace(base, strength=strength), reference)
        adjusted = engine.apply(features.loc[cal_index], scores.loc[cal_index]).adjusted_anomaly_score
        at_threshold = metrics(y_cal, adjusted, threshold=threshold)
        at_capacity = metrics(y_cal, adjusted, review_count=capacity, transaction_ids=ids_cal)
        rows.append({
            "strength": float(strength),
            "at_frozen_threshold": at_threshold,
            "at_equal_capacity": at_capacity,
            "recall_drop": baseline_threshold["recall"] - at_threshold["recall"],
            "false_positives_removed": baseline_threshold["fp"] - at_threshold["fp"],
            "detected_fraud_lost": baseline_threshold["tp"] - at_threshold["tp"],
            "relative_detected_fraud_loss": ((baseline_threshold["tp"] - at_threshold["tp"]) / baseline_threshold["tp"]
                                             if baseline_threshold["tp"] else None),
            "capacity_tp_gain": at_capacity["tp"] - baseline_capacity["tp"],
        })
    # Teslim edilen secim: eski olcut.
    eligible = [r for r in rows if r["recall_drop"] <= setup["maximum_calibration_recall_drop"] + 1e-12]
    original = min(eligible, key=lambda r: (-r["false_positives_removed"], r["recall_drop"], r["strength"]))
    # Dogru buldugum olcut: sabit kapasitede yakalanan fraud, sonra AP, sonra kucuk guc.
    capacity_first = max(rows, key=lambda r: (r["capacity_tp_gain"], r["at_equal_capacity"]["average_precision"], -r["strength"]))
    summary = {
        "purpose": "Post-hoc comparison of two selection criteria on the same calibration segment.",
        "not_a_preregistration": "The capacity-first criterion was written after seeing the original result. This file "
                                 "compares criteria; it is not independent evidence for the criterion.",
        "calibration_rows": len(cal_index), "frozen_threshold": threshold, "equal_capacity_reviews": capacity,
        "baseline_at_threshold": baseline_threshold, "baseline_at_equal_capacity": baseline_capacity,
        "candidates": rows,
        "original_criterion": {"rule": "Maximise removed false positives subject to recall drop <= "
                                      f"{setup['maximum_calibration_recall_drop']} at the frozen threshold.",
                              "selected_strength": original["strength"],
                              "shipped_strength": frozen["selected_context"]["strength"],
                              "flaw": "The objective followed the brief, which asks for false-positive reduction. The "
                                      "constraint did not: an absolute recall tolerance permits losing a large share of "
                                      "the fraud that was actually being caught, and the equal-capacity comparison was "
                                      "run after the selection instead of before."},
        "capacity_criterion": {"rule": "Maximise fraud caught at an equal review capacity, then average precision, "
                                      "then the smaller strength.",
                               "selected_strength": capacity_first["strength"],
                               "capacity_tp_gain": capacity_first["capacity_tp_gain"]},
        "criteria_agree": original["strength"] == capacity_first["strength"],
        "serving_default": "raw",
        "frozen_outputs_written": False,
        "test_labels_used": False,
        "input_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                         ("config/context.json", "config/context_evaluation.json", "scripts/reselect_context_strength.py")},
    }
    folder = ROOT / "artifacts/context_criterion"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render(summary, setup)
    print(json.dumps({"original_criterion_picks": original["strength"], "capacity_criterion_picks": capacity_first["strength"],
                      "shipped": frozen["selected_context"]["strength"], "agree": summary["criteria_agree"]}, indent=2), flush=True)


def conclusion(summary):
    """Sonuc cumlesi, iki olcutun ayni gucu secip secmesine degil kazanc sayilarina bakar.

    Onceden yalnizca `criteria_agree` kontrol ediliyordu. Iki olcut farkli guc
    secip ikisi de esit kapasitede pozitif kazanc verse bile "verideki bir
    kazanim degil" yaziyordu; bu gecerli bir girdide yanlis bir cikarim.
    """
    shipped = summary["original_criterion"]["shipped_strength"]
    rows = {row["strength"]: row for row in summary["candidates"]}
    shipped_gain = rows[shipped]["capacity_tp_gain"] if shipped in rows else None
    best_gain = summary["capacity_criterion"]["capacity_tp_gain"]
    selected = summary["capacity_criterion"]["selected_strength"]
    if summary["criteria_agree"]:
        return (f"İki ölçüt de {selected} gücünü seçiyor, yani bu veride ölçütün eksik kurulması sonucu "
                "değiştirmemiş olurdu.")
    if shipped_gain is not None and shipped_gain > 0:
        return (f"İki ölçüt farklı güç seçiyor, ama teslimdeki {shipped} gücü eşit kapasitede {shipped_gain:+d} fraud "
                f"kazandırıyor. Dolayısıyla \"verideki bir kazanım yok\" diyemem; ölçüt değişimi daha iyi bir gücü "
                f"({selected}, {best_gain:+d}) işaret ediyor, mevcut gücü geçersiz kılmıyor.")
    if best_gain > 0:
        return (f"Teslimdeki {shipped} gücü eşit kapasitede {shipped_gain:+d} fraud farkı veriyor, yani kazandırmıyor. "
                f"Kapasite ölçütünün seçtiği {selected} gücü ise {best_gain:+d} kazandırıyor: ölçüt değişimi başka bir "
                "gücü işaret ediyor, context fikrinin tamamını değil.")
    return (f"Hiçbir aday eşit kapasitede kazandırmıyor; en iyisi {selected} gücü ve onun farkı da {best_gain:+d}. "
            f"Teslimdeki {shipped} gücü {shipped_gain:+d} veriyor, yani sıfırdan farklı her güç aynı kapasitede daha az "
            "fraud yakalıyor. Bu durumda context'in devreye girmesinin sebebi verideki bir kazanım değil, ölçütün "
            "eksik kurulmasıydı. Servis varsayılanının `raw` olması bu sonuçla tutarlı; farkı artık \"audit sonucunu "
            "beğenmedim\" diye değil, \"ölçütü eksik kurmuşum\" diye söylüyorum.")


def render(summary, setup):
    (ROOT / "reports/context_criterion.md").write_text(report_lines(summary, setup), encoding="utf-8")


def report_lines(summary, setup):
    lines = ["# Seçim ölçütünü yeniden kurmak", "",
             "Context deneyine başlarken başarı ölçütünü şöyle kurmuştum: sabit eşikte false positive azalsın, recall en "
             "fazla 1 yüzde puan düşsün. Teslimdeki 0.5 gücünü seçen de bu ölçüttü.", "",
             "Hedef tarafı case'in istediği şeydi: azaltılacak olan false positive. Eksik olan kısıt tarafı. Tolerans "
             "mutlak puan üzerinden tanımlı; baseline recall %8 civarında olduğu için 1 puanlık izin, yakalanan "
             "fraud'un yaklaşık onda birini kaybetmek anlamına geliyor. Ölçüt bu oranı hiç görmüyor. Birincil ölçüt "
             "verilen inceleme kapasitesinde yakalanan fraud olmalıydı: kapasite sabitken daha az alarm üretmek kazanç "
             "değildir, daha çok fraud yakalamak kazançtır. O karşılaştırmayı yaptım ama seçimden sonra.", "",
             "Aşağıdaki yeniden seçimde ayrı bir göreli kayıp sınırı **uygulamıyorum.** Kapasite sabitken TP farkı "
             "kaybı da kazancı da aynı sayıda ölçüyor, ikinci bir tolerans gereksiz olurdu. Göreli kaybı tabloda aday "
             "başına raporluyorum ama eleme filtresi olarak kullanmıyorum; seçim sırası kapasitedeki TP farkı, sonra "
             "AP, sonra küçük güç. Bunu ayrıca yazıyorum çünkü \"kısıtı düzelttim\" demek, kodda olmayan bir filtre "
             "ima eder.", "",
             "Bunu geçmişe dönük düzeltmiş gibi yazmıyorum. Aşağıdaki tablo, ilk seçimin kullandığı aynı calibration "
             f"yarısında ({summary['calibration_rows']:,} işlem) iki ölçütün ne seçtiğini yan yana koyuyor. Yeni ölçütün "
             "bağımsız doğrulaması değil; ölçülen şey ölçütün kararı nasıl değiştirdiği.", "",
             f"Eşik train skorlarından sabitlenmiş: {summary['frozen_threshold']:.8f}. Eşit kapasite karşılaştırması ham "
             f"skorun bu eşikte ürettiği {summary['equal_capacity_reviews']:,} incelemeyi kullanıyor.", "",
             "| Güç | Sabit eşikte kaldırılan FP | Kaybedilen fraud | Kaybın oranı | Recall düşüşü | Eşit kapasitede fraud farkı |",
             "|---|---:|---:|---:|---:|---:|"]
    for row in summary["candidates"]:
        share = "n/a" if row["relative_detected_fraud_loss"] is None else f"%{row['relative_detected_fraud_loss']*100:.2f}"
        lines.append(f"| {row['strength']} | {row['false_positives_removed']:+d} | {row['detected_fraud_lost']:+d} | "
                     f"{share} | {row['recall_drop']*100:.3f} puan | {row['capacity_tp_gain']:+d} |")
    original, capacity = summary["original_criterion"], summary["capacity_criterion"]
    lines += ["", "## İki ölçüt ne seçiyor", "",
              f"- İlk ölçüt (teslimdeki): **{original['selected_strength']}**. Teslimde çalışan güç de "
              f"{original['shipped_strength']}, yani tablo teslimi yeniden üretiyor.",
              f"- Kapasite ölçütü: **{capacity['selected_strength']}**, eşit kapasitede fraud farkı "
              f"{capacity['capacity_tp_gain']:+d}.", ""]
    lines += ["", conclusion(summary)]
    lines += ["", "## Sınırlar", "",
              "Bu hesap calibration yarısını ikinci kez kullanıyor. Yeni ölçütün doğruluğunu kanıtlamaz, yalnızca ilk "
              "ölçütün bu veride nasıl bir karar ürettiğini gösterir. Dondurulmuş çıktıların hiçbirine yazmıyor: "
              "`artifacts/context/selected_config.json`, `data/processed/context_scores.parquet` ve final artifact'ları "
              "değişmedi. Final test etiketleri okunmadı.", "",
              f"Aday güçler ({', '.join(str(s) for s in setup['candidate_strengths'])}) ilk deneyde etiket görülmeden "
              "yazılmıştı; bu koşu aynı listeyi kullanıyor, yeni aday eklemiyor.", "",
              "Yeniden çalıştırma: `python scripts/reselect_context_strength.py`. Çıktı: "
              "`artifacts/context_criterion/evaluation.json`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
