"""İndirim gücünü erken validation'da seçer, sonraki yarıda kontrol eder."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_case.context import CONTEXT_RULES, ContextConfig, ContextEngine
from fraud_case.evaluation import metrics, paired_effect


def validation_sections(validation: pd.DataFrame, fraction: float):
    if not 0 < fraction < 1 or len(validation) < 2:
        raise ValueError("Invalid validation calibration fraction")
    if not validation.TransactionDT.is_monotonic_increasing:
        raise ValueError("Validation must be chronological")
    boundary = validation.TransactionDT.iloc[int(len(validation) * fraction)]
    calibration = validation.TransactionDT < boundary
    if not calibration.any() or calibration.all():
        raise ValueError("Need nonempty, separate timestamp groups for calibration and audit")
    return calibration, ~calibration, float(boundary)


def select_strength(labels, raw_scores, candidate_scores, threshold, max_recall_drop):
    baseline = metrics(labels, raw_scores, threshold=threshold)
    if baseline["recall"] is None:
        raise ValueError("Calibration needs fraud observations")
    rows = []
    for strength, scores in candidate_scores.items():
        result = metrics(labels, scores, threshold=threshold)
        loss = baseline["recall"] - result["recall"]
        rows.append({"strength": float(strength), "metrics": result, "recall_drop": loss,
                     "false_positives_removed": baseline["fp"] - result["fp"],
                     "eligible": loss <= max_recall_drop + 1e-12})
    eligible = [row for row in rows if row["eligible"]]
    if not eligible:
        raise ValueError("No eligible context candidate; include a zero-strength baseline")
    selected = min(eligible, key=lambda row: (-row["false_positives_removed"], row["recall_drop"], row["strength"]))
    return selected["strength"], rows


def compare(frame, threshold):
    labels = frame.isFraud.to_numpy()
    baseline = metrics(labels, frame.raw_anomaly_score, threshold=threshold)
    adjusted = metrics(labels, frame.adjusted_anomaly_score, threshold=threshold)
    count = baseline["alerts"]
    budget = {name: metrics(labels, frame[column], review_count=count, transaction_ids=frame.TransactionID)
              for name, column in (("baseline", "raw_anomaly_score"), ("adjusted", "adjusted_anomaly_score"))}
    return {"baseline": baseline, "adjusted": adjusted, "same_review_budget": budget,
            "paired_effect": paired_effect(labels, frame.raw_anomaly_score, frame.adjusted_anomaly_score, threshold)}


def decision_boundary(result, prevention_rate=.5):
    """Basa bas orani tek bir sayi degil; hangi maliyet modelinde ne oldugu.

    Context yalnizca kazanilan seyin degeri kaybedilen seyin degerinden buyukse
    kazandirir. "Kazanilan sey" maliyet modeline gore degisiyor: sadece yanlis
    alarm mi, her inceleme mi. Kacirilan fraud tarafinda da her yakalanan
    fraud'un gercekten onlendigi varsayimi var; onlenme orani 1'den kucukse
    esik o oranda yukseliyor. Sayilari veren sirket, esigi buradan okur.
    """
    effect = result["paired_effect"]
    net_false_positives = effect["removed_false_positives"] - effect["added_false_positives"]
    net_true_positives = effect["lost_true_positives"] - effect["added_true_positives"]
    removed_reviews = result["baseline"]["alerts"] - result["adjusted"]["alerts"]
    if net_true_positives <= 0:
        return {"status": "no_fraud_lost", "net_false_positives_removed": net_false_positives,
                "removed_reviews": removed_reviews}
    ratio = net_false_positives / net_true_positives
    return {
        "status": "ok", "net_false_positives_removed": net_false_positives,
        "net_true_positives_lost": net_true_positives, "removed_reviews": removed_reviews,
        "models": {
            "false_positive_cost_only": {
                "quantity": "C_missed_fraud / C_false_alarm", "break_even": ratio,
                "assumption": "Only false alarms cost anything; a true alarm's review is free."},
            "every_review_costs": {
                "quantity": "C_missed_fraud / C_review", "break_even": removed_reviews / net_true_positives,
                "assumption": "Every review costs the same, whatever its outcome."},
            "partial_prevention": {
                "quantity": "C_missed_fraud / C_false_alarm", "break_even": ratio / prevention_rate,
                "prevention_rate": prevention_rate,
                "assumption": "Only this share of caught fraud is actually prevented; the rest is lost anyway."},
        },
        "reading": "Context pays only while the listed quantity stays BELOW its break-even value.",
        "not_modelled": ["Transaction amounts: a lost fraud is treated as one unit, not as its value.",
                         "Investigator capacity limits and queueing.",
                         "Downstream effects of a false alarm on the customer."],
    }


def run(root: Path):
    base = ContextConfig(**json.loads((root / "config/context.json").read_text(encoding="utf-8")))
    evaluation = json.loads((root / "config/context_evaluation.json").read_text(encoding="utf-8"))
    features = pd.read_parquet(root / "data/processed/features.parquet")
    scores = pd.read_parquet(root / "data/processed/scores.parquet")
    times = pd.read_parquet(root / "data/processed/transactions.parquet", columns=["TransactionID", "TransactionDT", "ProductCD"])
    if not features.TransactionID.equals(times.TransactionID):
        raise ValueError("Feature/time alignment mismatch")
    features["TransactionDT"] = times.TransactionDT
    features["ProductCD"] = times.ProductCD
    reference = ContextEngine(base).fit_product_context(features.loc[features["split"].eq("train")]).product_reference
    # Predicate pushdown - final test etiketleri belleğe hiç gelmiyor
    labels = pd.read_parquet(root / "data/processed/transactions.parquet", columns=["TransactionID", "isFraud"], filters=[("split", "=", "validation")])
    validation = features.loc[features["split"].eq("validation")]
    cal_mask, audit_mask, boundary = validation_sections(validation, evaluation["validation_calibration_fraction"])
    cal_index, audit_index = validation.index[cal_mask], validation.index[audit_mask]
    y = labels.set_index("TransactionID").reindex(validation.TransactionID).set_axis(validation.index)["isFraud"]
    if y.isna().any():
        raise ValueError("Validation label alignment mismatch")
    train_scores = scores.loc[scores["split"].eq("train"), "raw_anomaly_score"]
    fraction = evaluation["training_review_fraction"]
    threshold = float(train_scores.quantile(1 - fraction, interpolation="higher"))
    print(f"Frozen train-derived threshold: {threshold:.8f}; calibration={len(cal_index):,}, audit={len(audit_index):,}", flush=True)
    candidates = {}
    for strength in evaluation["candidate_strengths"]:
        engine = ContextEngine(replace(base, strength=strength), reference)
        candidates[strength] = engine.apply(features.loc[cal_index], scores.loc[cal_index]).adjusted_anomaly_score
    selected, search = select_strength(y.loc[cal_index], scores.loc[cal_index].raw_anomaly_score,
                                       candidates, threshold, evaluation["maximum_calibration_recall_drop"])
    config = replace(base, strength=selected)
    engine = ContextEngine(config, reference)
    print(f"Selected strength on calibration: {selected}; applying frozen context policy ...", flush=True)
    adjusted = engine.apply(features, scores)
    audited = adjusted.loc[audit_index].copy()
    audited["isFraud"] = y.loc[audit_index]
    result = compare(audited, threshold)
    sensitivity = []
    for rate in evaluation["sensitivity_review_fractions"]:
        cutoff = float(train_scores.quantile(1 - rate, interpolation="higher"))
        comparison = compare(audited, cutoff)
        sensitivity.append({"train_review_fraction": rate, "threshold": cutoff, **comparison})
    ablation = {}
    for name in CONTEXT_RULES:
        one = engine.apply(features.loc[audit_index], scores.loc[audit_index], enabled_rules={name})
        one["isFraud"] = y.loc[audit_index]
        ablation[name] = compare(one, threshold)
    coverage = {}
    for name, mask in (("all_four_layers", audited.available_weight.ge(1 - 1e-12)), ("partial_layers", audited.available_weight.lt(1 - 1e-12))):
        if mask.any():
            coverage[name] = compare(audited.loc[mask], threshold)
    baseline, final = result["baseline"], result["adjusted"]
    audit_recall_drop = baseline["recall"] - final["recall"]
    summary = {
        "evaluation": evaluation, "selected_context": asdict(config), "threshold": threshold,
        "product_reference": reference,
        "audit_status": "Later-validation development check. Baseline aggregate metrics were observed in an earlier no-op iteration; final test remains untouched.",
        "validation_boundary_seconds": boundary, "calibration_rows": len(cal_index), "audit_rows": len(audit_index),
        "calibration_candidates": search, "audit": result, "audit_sensitivity": sensitivity,
        "audit_ablation": ablation, "audit_by_coverage": coverage,
        "audit_recall_drop": audit_recall_drop,
        "audit_acceptance": final["fp"] < baseline["fp"] and audit_recall_drop <= evaluation["maximum_calibration_recall_drop"],
        "fixed_budget_ranking_improved": result["same_review_budget"]["adjusted"]["tp"] > result["same_review_budget"]["baseline"]["tp"],
        # Kaldirilan FP ile kaybedilen TP'nin orani kararin donum noktasi: context ancak
        # kacirilan bir fraud bu kadar yanlis alarmdan UCUZSA kazandirir.
        "break_even_fn_per_fp": (result["paired_effect"]["removed_false_positives"]
                                 / result["paired_effect"]["lost_true_positives"]
                                 if result["paired_effect"]["lost_true_positives"] else None),
        "decision_boundary": decision_boundary(result),
        "relative_fp_reduction": (baseline["fp"] - final["fp"]) / baseline["fp"] if baseline["fp"] else None,
        "relative_detected_fraud_loss": (baseline["tp"] - final["tp"]) / baseline["tp"] if baseline["tp"] else None,
        "external_context": "No verified trust or weekend schedule supplied; calendar reference absent.",
        "test_labels_used": False,
        "audit_policy_activity": {name: int(audited[f"context_match_{name}"].sum()) for name in CONTEXT_RULES},
        "audit_adjusted_rows": int(audited.context_reduction.gt(0).sum()),
    }
    output = root / "artifacts/context"
    output.mkdir(parents=True, exist_ok=True)
    for filename in ("config/context.json", "config/context_evaluation.json", "data/processed/transactions.parquet", "data/processed/features.parquet", "data/processed/scores.parquet"):
        with (root / filename).open("rb") as stream:
            summary.setdefault("input_sha256", {})[filename] = hashlib.file_digest(stream, "sha256").hexdigest()
    (output / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "selected_config.json").write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
    (output / "product_reference.json").write_text(json.dumps(reference, indent=2) + "\n", encoding="utf-8")
    adjusted.to_parquet(root / "data/processed/context_scores.parquet", index=False)
    changed = audited.loc[audited.context_reduction.gt(0)]
    example_index = changed.index[0] if len(changed) else audit_index[0]
    explanation = engine.explain(features.loc[[example_index]], scores.loc[[example_index]])
    (output / "explanation.json").write_text(json.dumps(explanation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render_report(root, summary)
    print(json.dumps({"selected_strength": selected, "threshold": threshold, "audit": result,
                      "audit_acceptance": summary["audit_acceptance"], "test_labels_used": False}, indent=2), flush=True)


def percentage(value):
    return "n/a" if value is None else f"%{100 * value:.3f}"


def render_report(root, summary):
    baseline = summary["audit"]["baseline"]
    adjusted = summary["audit"]["adjusted"]
    effect = summary["audit"]["paired_effect"]
    boundary = summary["decision_boundary"]
    budget = summary["audit"]["same_review_budget"]
    lines = ["# Context Adjust Engine", "", "## Deney tasarımı", "",
             "Context, anomali katmanlarının ayrı katkılarını gerekçeli ve sınırlı şekilde azaltır. Bilinen yeni ilişki, "
             "güçlü multivariate/entity sapması veya yüksek saatlik velocity varsa indirim bloke edilir. Aynı katmana uyan "
             "kuralların en büyük indirimi alınır; eşitlikte dosyadaki sıra korunur. Toplam indirim en fazla 0.05 puan ve "
             "raw skorun %15'idir.", "",
             "Bu kurallar operasyonel hipotezlerdir; gerçek şirket politikası olarak sunulmaz. İndirim gücü adayları (0, "
             "0.25, 0.5, 1) calibration değerlendirmesinden önce tanımlandı. Validation'ın ilk yarısında, recall kaybı en "
             "fazla 1 yüzde puan olmak üzere en çok false positive azaltan güç seçildi. Eşitlikte daha az recall kaybı, "
             "sonra daha küçük güç tercih edilir. Sonraki validation yarısı güç seçiminde kullanılmadı. İlk no-op "
             "denemesinde bu yarının toplam baseline metrikleri görülmüş olduğundan sonuç bir geliştirme kontrolüdür; hiç "
             "görülmemiş final holdout olarak sunulmaz. Final test etiketleri okunmadı.", "",
             f"- Calibration: {summary['calibration_rows']:,} satır; audit: {summary['audit_rows']:,} satır.",
             f"- Aynı saniye gruplarını ayırmayan sınır: {summary['validation_boundary_seconds']:.0f}.",
             f"- Eşik: {summary['threshold']:.8f}. Train raw skorlarının üst yaklaşık %5'i için seçildi; iş kapasitesi verilmediğinden %5 deney varsayımıdır.",
             f"- Calibration'da seçilen indirim gücü: {summary['selected_context']['strength']}.", "",
             "## Context kuralları", "", "| Kural | Etkilenen katman | Koşul | Azaltım |", "|---|---|---|---|",
             "| İşlem türü için alışıldık tutar | Column | Global tutar rank >=0.90; kendi ProductCD grubunun train p10–p90 "
             "aralığında; en az 1000 train gözlemi; guard yok | Katman katkısının %10'u × seçilen güç |",
             "| Sık ve alışıldık faaliyet | Temporal | En az 20 geçmiş işlem, 7 gün, alışıldık tutar/ilişkiler ve "
             "desteklenen saat fazı | Katman katkısının %20'si × seçilen güç |",
             "| Entity için alışıldık yüksek tutar | Column | Yukarıdaki yerleşik davranış ve global tutar yüzdeliği >=0.90 | %15 × güç |",
             "| İş saatleri | Temporal | Açık takvim, hafta içi 09–17 ve istikrarlı entity davranışı | %10 × güç |",
             "| Beklenen hafta sonu faaliyeti | Temporal | Açık takvim, hafta sonu, işlemden önceki dış çalışma takvimi ve "
             "istikrarlı davranış | %10 × güç |",
             "| Doğrulanmış güven | Multivariate | İşlemden önceki kaynaklı güven kaydı ve istikrarlı davranış | %10 × güç |", "",
             "İstikrarlı davranışta amount/prior_mean 0.8–1.25 aralığında, entity_raw <=1 ve ürün ilişkisi bilinen "
             "olmalıdır. Bilinen yeni ürün/e-posta alanı/cihaz açıklaması indirimi bloke eder. Bilinmeyen e-posta/cihaz "
             "geçmişi güven kanıtı sayılmaz; kararın mevcut ürün/tutar geçmişine dayandığı kabul edilir.", "",
             "## Aynı eşikte: validation audit", "", "| Ölçüm | Raw | Context sonrası |", "|---|---:|---:|"]
    for name, label in (("alerts", "Alarm"), ("tp", "Yakalanan fraud (TP)"), ("fp", "Yanlış alarm (FP)"), ("fn", "Kaçırılan fraud (FN)"), ("tn", "Doğru normal (TN)")):
        lines.append(f"| {label} | {baseline[name]:,} | {adjusted[name]:,} |")
    for name, label in (("precision", "Precision"), ("recall", "Recall"), ("false_positive_rate", "False positive rate"), ("average_precision", "Average Precision (AP)")):
        lines.append(f"| {label} | {percentage(baseline[name])} | {percentage(adjusted[name])} |")
    lines += ["", f"Aynı işlemler üzerinde {effect['removed_false_positives']} yanlış alarm kaldırıldı; {effect['lost_true_positives']} fraud alarmı da kaldırıldı. Recall farkı {summary['audit_recall_drop'] * 100:.3f} yüzde puan kayıptır. FP sayısı azalması ile FPR azalması farklı ölçümlerdir; FPR paydası gerçek normal işlem sayısıdır.", "",
              f"Bu revizyonun sayısal audit ölçütü (FP azalması ve recall kaybı <=1 yüzde puan): **{'geçti' if summary['audit_acceptance'] else 'geçmedi'}**. Güç seçimi sonrası audit sonucuna göre parametre değiştirilmedi. Recall toleransı deney varsayımıdır; şirketin kabul ettiği kayıp bütçesi değildir. Bu tek zamansal kesitteki sonuç, üretim etkisi veya istatistiksel anlamlılık garantisi değildir.", "",
              f"FP sayısındaki göreli azalma {percentage(summary['relative_fp_reduction'])}; baseline'ın yakaladığı fraud sayısındaki göreli kayıp {percentage(summary['relative_detected_fraud_loss'])}. Küçük bir mutlak recall farkı, yakalanan fraud'ların kayda değer bölümünü kaybetmek anlamına gelebilir.", "",
              "### Kararın dönüm noktası", "",
              f"Maliyet verisi elimde yok, ama kararı verilebilir hale getiren orana ihtiyacım da yok: "
              f"{effect['removed_false_positives']} yanlış alarm kaldırdım ve {effect['lost_true_positives']} fraud "
              f"kaybettim. Tek bir başa baş sayısı vermek yetersiz olurdu, çünkü sayı kayıp fonksiyonunu nasıl "
              "kurduğunuza bağlı. Üç açık model:", "",
              "| Maliyet modeli | Varsayım | Context kazandıran koşul |", "|---|---|---|",
              *([f"| Sadece yanlış alarm maliyetli | Doğru alarmın incelemesi bedava | kaçırılan fraud / yanlış alarm "
                 f"< **{boundary['models']['false_positive_cost_only']['break_even']:.2f}** |",
                 f"| Her inceleme maliyetli | Sonucu ne olursa olsun her inceleme aynı maliyet | kaçırılan fraud / "
                 f"inceleme < **{boundary['models']['every_review_costs']['break_even']:.2f}** |",
                 f"| Yakalananın yalnızca %{boundary['models']['partial_prevention']['prevention_rate']*100:.0f}'i "
                 f"önlenebiliyor | Kalanı zaten kaybedilecekti | kaçırılan fraud / yanlış alarm < "
                 f"**{boundary['models']['partial_prevention']['break_even']:.2f}** |"]
                if boundary["status"] == "ok" else
                ["| - | Bu koşuda hiç fraud kaybedilmedi, oran tanımsız | - |"]), "",
              "Bu koşulların sağlanmasını beklemiyorum, ama bu bir ölçüm değil beklenti: bana maliyet verisi "
              "verilmedi, dolayısıyla oranın gerçekte nerede olduğunu bu çalışmadan bilmiyorum. Beklentimin dayanağı "
              "şu gerekçe: bir yanlış alarmın maliyeti birkaç dakikalık inceleme, kaçırılan fraud'un maliyeti işlem "
              "tutarı artı geri ödeme. Gerekçeyi bulgu gibi sunmuyorum; şirket kendi iki maliyetini yukarıdaki "
              "eşiklere koyup kararı kendisi verir.", "",
              "Modellemediğim şeyler: işlem tutarı (kaybedilen fraud'u bir birim sayıyorum, tutarını değil), "
              "inceleme kapasitesinin kuyruk etkisi, yanlış alarmın müşteri tarafındaki maliyeti. Bu üçü olmadan "
              "kesin bir ekonomik sonuç yazmam doğru olmazdı.", "",
              "### Seçim ölçütündeki eksik", "",
              "Yukarıdaki güç seçimini kuran ölçüt şuydu: sabit eşikte false positive azalt, recall kaybını 1 yüzde "
              "puanla sınırla. Hedef tarafı case'in istediği şeydi, azaltılacak olan false positive. Eksik olan kısıt "
              "tarafı: kaybı mutlak puanla sınırlamak, baseline recall %8 civarındayken yakalanan fraud'un onda birini "
              "kaybetmeye izin veriyor. Birincil ölçüt sabit inceleme kapasitesinde yakalanan fraud olmalıydı; "
              "kapasite sabitken TP farkı kaybı da kazancı da aynı sayıda ölçtüğü için ayrı bir kayıp toleransına "
              "gerek kalmıyor. Aynı calibration yarısında o kurulumla yeniden seçim koştum: seçim **0**, "
              "yani context hiç devreye girmezdi ([ölçüt karşılaştırması](context_criterion.md)). Sıfırdan farklı her "
              "güç eşit kapasitede daha az fraud yakalıyor. Yani context'in devreye girmesinin nedeni verideki bir "
              "kazanım değil, ölçütün eksik kurulmasıydı. Aynı kapasite karşılaştırmasını seçimden sonra değil önce "
              "yapmam gerekirdi.", "",
              "Bu sonucu servise de yazdım. Servisin varsayılan context profili `raw`: indirim uygulanmaz, "
              "`adjusted_anomaly_score` raw skora eşit çıkar. Buradaki davranış `behavior_context` profiliyle "
              "açıkça istenirse çalışır, `product_risk` ayrı bir geliştirme adayıdır. Eşleşen kural listesi üç "
              "profilde de raporlanır; kapattığım şey kuralın görünürlüğü değil skora etkisi. Varsayılanın gerçekten "
              "indirim uygulamadığını `scripts/verify_context_api.py` altı senaryoda ve 64 kayıtlı işlemde ölçer, "
              "`tests/test_api.py` de aynı sözleşmeyi test eder. Aksi halde bu bölüm 'açmamak gerekir' derken kod "
              "açık bırakmış olurdu.", "",
              "## Aynı inceleme kapasitesi", "", f"İki yöntem de audit üzerinde tam {baseline['alerts']:,} işlemi incelemeye gönderirse (skor eşitliğinde TransactionID sırası):", "",
              "| Ölçüm | Raw | Context sonrası |", "|---|---:|---:|"]
    for name, label in (("tp", "Yakalanan fraud"), ("fp", "Yanlış alarm")):
        lines.append(f"| {label} | {budget['baseline'][name]:,} | {budget['adjusted'][name]:,} |")
    lines += ["", "Bu karşılaştırma, yalnızca daha az alarm üretmekten doğan görünür iyileşmeyi sıralama değişiminden ayırır. Average Precision bütün sıralamayı değerlendirir; bir eşikte FP azaltmak AP'nin de yükseldiği anlamına gelmez.", "",
              f"Sabit bütçede yakalanan fraud artışı: **{'var' if summary['fixed_budget_ranking_improved'] else 'yok'}**. Dolayısıyla bu çalıştırmanın context indirimi genel bir model iyileşmesi olarak sunulmaz. Alarm maliyeti ve kaçırılan fraud maliyeti verilmeden işletme açısından tercih kararı çıkarılamaz.", "",
              "## Tek kural etkileri (audit, aynı eşik)", "", "| Tek aktif kural | Eşleşen işlem | Kaldırılan FP | Kaybedilen TP |", "|---|---:|---:|---:|"]
    for name, comparison in summary["audit_ablation"].items():
        paired = comparison["paired_effect"]
        lines.append(f"| {name} | {summary['audit_policy_activity'][name]:,} | {paired['removed_false_positives']} | {paired['lost_true_positives']} |")
    lines += ["", "Tek kural etkileri üst üste toplanamaz; aynı işlem birden fazla koşula uyabilir ve ortak cap vardır.", "",
              "## İşlem türü bağlamı", "",
              "ProductCD kodlarının gerçek iş anlamı bilinmez. Her ürün kodunda en az 1000 train gözlemiyle öğrenilen p10–p90 "
              "tutar aralığı kullanılır. Eşleşmeyen veya desteklenmeyen yeni ürün koduna indirim verilmez. Bu kural tutar "
              "sinyalinin yalnızca ilgili katkısını azaltır; diğer katmanlar korunur. Referans aralıkları train dışındaki "
              "kayıtlarla güncellenmez. İlk entity-only denemesi eşik üzerinde hiç FP azaltmadı; ürün bağlamı bu tespit "
              "sonrası, calibration feature'ları üzerinde geliştirildi.", "",
              "## Eksik dış bağlam", "",
              "IEEE-CIS başlangıç takvimi, gerçek müşteri güven listesi ve çalışma takvimi sağlamaz. Bu çalıştırmada "
              "business-hours, weekend ve trust dalları veri uydurularak etkinleştirilmedi. Üç dal sentetik senaryo "
              "testlerinde doğrulandı; gerçek FP etkileri burada ölçülemedi. Sık entity, trusted entity olarak etiketlenmedi.", "",
              "Dış bağlam TransactionID ile eşlenir. Trusted flag için trust_source ve trust_observed_at; hafta sonu planı "
              "için schedule_source ve schedule_observed_at gerekir. Zamanlar TransactionDT ile aynı saniye referansında "
              "olmalıdır. Kaynak eksikse hata; kayıt işlemden sonra veya aynı saniyede gözlenmişse indirim yoktur. Alanların "
              "doğruluğu sağlayan sisteme aittir; bu prototip kaynağı harici bir kurumdan doğrulamaz.", "",
              "## Çıktılar", "", "- `data/processed/context_scores.parquet`: orijinal skorlar, kural eşleşmeleri, katman indirimleri, nedenler ve adjusted_anomaly_score.",
              "- `artifacts/context/evaluation.json`: calibration araması, audit, sabit bütçe, eşik duyarlılığı, kapsama "
              "grupları ve girdi SHA-256 değerleri.",
              "- `artifacts/context/selected_config.json`: yalnızca calibration'da seçilen sabit yapılandırma.",
              "- `artifacts/context/explanation.json`: audit içindeki bir işlemin gerçek indirim açıklaması.", "",
              "![Context karşılaştırması](figures/context_evaluation.png)", ""]
    (root / "reports/context_adjustment.md").write_text("\n".join(lines), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, comparison, title in ((axes[0], summary["audit"], "Same threshold"), (axes[1], budget, "Same review budget")):
        x = np.arange(2)
        ax.bar(x - .18, [comparison["baseline"]["fp"], comparison["baseline"]["tp"]], width=.36, label="Raw", color="#256580")
        ax.bar(x + .18, [comparison["adjusted"]["fp"], comparison["adjusted"]["tp"]], width=.36, label="Context", color="#ae694c")
        ax.set_xticks(x, ["False positives", "Detected fraud"])
        ax.set(title=title, ylabel="Transactions")
        ax.legend(frameon=False)
        for bars in ax.containers:
            ax.bar_label(bars, padding=2, fontsize=9)
        ax.margins(y=.18)
    fig.suptitle(f"Later validation audit: {summary['audit_rows']:,} transactions")
    fig.savefig(root / "reports/figures/context_evaluation.png", dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
