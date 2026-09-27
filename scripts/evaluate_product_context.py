"""Train etiketlerinden ürün riski context'ini değerlendirir, test'e dokunmadan."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_case.context import CONTEXT_RULES, ContextConfig, ContextEngine
from fraud_case.evaluate_context import validation_sections
from fraud_case.evaluation import metrics
from fraud_case.product_context import ProductRiskContextEngine

ROOT = Path(__file__).resolve().parents[1]


def run():
    folder = ROOT / "artifacts/product_context"
    folder.mkdir(parents=True, exist_ok=True)
    plan = {"case_scope": "Step 6: transaction-type risk profiles, explicitly suggested in the case.",
            "strengths": [0., .025, .05, .10], "prior_pseudocount": 1000, "minimum_product_support": 1000,
            "rule": "Add strength * clipped log(product smoothed training fraud rate / global training fraud rate). Clamp log ratio to [-1,1] and final score to [0,1]. Existing strong-signal guard blocks negative adjustments. Unknown/unsupported product makes no risk adjustment. Other existing behavior/calendar/trust rules remain; product-typical-amount discount is replaced.",
            "selection": "Early validation in two chronological blocks, 5% review budget each: maximize minimum TP gain versus original raw, then total TP, then AP, then smaller strength.",
            "training_labels": "Product profile uses train labels. Assume these historical labels are available before scoring validation; no fraud-report delay is supplied.",
            "validation_status": "Previously seen development data; later half not used to select this candidate. Not an independent holdout.",
            "test_labels_read": False, "production_promotion": False}
    (folder / "plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    filters = [("split", "in", ["train", "validation"])]
    source = pd.read_parquet(ROOT / "data/processed/transactions.parquet", columns=["TransactionID", "TransactionDT", "ProductCD", "isFraud", "split"], filters=filters).reset_index(drop=True)
    features = pd.read_parquet(ROOT / "data/processed/features.parquet", filters=filters).reset_index(drop=True)
    scores = pd.read_parquet(ROOT / "data/processed/scores.parquet", filters=filters).reset_index(drop=True)
    assert source.TransactionID.equals(features.TransactionID) and source.TransactionID.equals(scores.TransactionID)
    train = source["split"].eq("train")
    global_rate = float(source.loc[train, "isFraud"].mean())
    profile = source.loc[train].groupby("ProductCD", observed=True).isFraud.agg(["count", "sum"])
    profile["smoothed_rate"] = (profile["sum"]+1000*global_rate)/(profile["count"]+1000)
    profile["log_relative_risk"] = np.log(profile.smoothed_rate/global_rate).clip(-1, 1)
    profile.loc[profile["count"].lt(1000), "log_relative_risk"] = 0.
    profiles = {str(key): {"count": int(row["count"]), "fraud_count": int(row["sum"]),
                          "smoothed_rate": float(row.smoothed_rate), "log_relative_risk": float(row.log_relative_risk)}
                for key, row in profile.iterrows()}
    features["TransactionDT"], features["ProductCD"] = source.TransactionDT, source.ProductCD
    v_index = features.index[~train]
    cal, audit, _ = validation_sections(features.loc[v_index], .5)
    cal_index, audit_index = cal.index[cal], audit.index[audit]
    first, second, _ = validation_sections(features.loc[cal_index], .5)
    blocks = [first.index[first], second.index[second]]
    old = json.loads((ROOT / "artifacts/context/evaluation.json").read_text())
    behavior = ContextEngine(ContextConfig(**old["selected_context"]), old["product_reference"]).apply(
        features, scores, enabled_rules=set(CONTEXT_RULES)-{"product_typical_amount"})
    relative = features.ProductCD.astype("string").map({key: v["log_relative_risk"] for key, v in profiles.items()}).fillna(0.)
    relative = relative.mask(behavior.context_guard_active & relative.lt(0), 0.)
    # Servis ürün riskini yalnızca train döneminden sonraki işlemlere uygular
    # (product_context.py: TransactionDT > training_end). Aynı kapıyı burada da
    # açıyorum. Açmazsam eşiği train satırlarından, servisin o satırlarda hiç
    # üretmediği bir skor dağılımı üzerinden hesaplamış olurum; deneydeki eşik
    # ile servisteki skor aynı fonksiyondan gelmez.
    training_end = float(source.loc[train, "TransactionDT"].max())
    relative = relative.where(features.TransactionDT.gt(training_end), 0.)
    def budget(values, index, fraction=.05):
        return metrics(source.loc[index, "isFraud"], values.loc[index], review_count=int(np.ceil(len(index)*fraction)),
                       transaction_ids=source.loc[index, "TransactionID"])
    baseline_blocks = [budget(scores.raw_anomaly_score, index) for index in blocks]
    candidates, arrays = [], {}
    for strength in plan["strengths"]:
        adjusted = (behavior.adjusted_anomaly_score + strength*relative).clip(0, 1)
        rows = [budget(adjusted, index) for index in blocks]
        threshold = float(adjusted.loc[train].quantile(.95, interpolation="higher"))
        row = {"strength": strength, "blocks": rows, "calibration": budget(adjusted, cal_index), "threshold": threshold,
               "minimum_block_tp_gain": min(a["tp"]-b["tp"] for a,b in zip(rows, baseline_blocks, strict=True))}
        candidates.append(row)
        arrays[strength] = adjusted
    selected = max(candidates, key=lambda r: (r["minimum_block_tp_gain"], r["calibration"]["tp"], r["calibration"]["average_precision"], -r["strength"]))
    adjusted = arrays[selected["strength"]]
    original_context = pd.read_parquet(ROOT / "data/processed/context_scores.parquet", columns=["TransactionID", "adjusted_anomaly_score"], filters=[("split", "=", "validation")]).set_index("TransactionID")
    legacy = source.TransactionID.map(original_context.adjusted_anomaly_score)
    sensitivity = {str(rate): {"raw": budget(scores.raw_anomaly_score, audit_index, rate),
                              "original_context": budget(legacy, audit_index, rate),
                              "product_context": budget(adjusted, audit_index, rate)} for rate in [.01, .05, .10]}
    same_threshold = {"raw": metrics(source.loc[audit_index, "isFraud"], scores.loc[audit_index, "raw_anomaly_score"], threshold=old["threshold"]),
                      "product_context": metrics(source.loc[audit_index, "isFraud"], adjusted.loc[audit_index], threshold=selected["threshold"])}
    policy = {"version": 1, "strength": selected["strength"], "global_train_rate": global_rate,
              "prior_pseudocount": plan["prior_pseudocount"], "minimum_product_support": plan["minimum_product_support"],
              "profiles": profiles, "threshold": selected["threshold"], "base_context_config": old["selected_context"],
              "product_reference": old["product_reference"], "status": "development_candidate_not_promoted",
              "training_end_transaction_dt": training_end}
    # Deneydeki formülü servisin kendi motoruna karşı ölçüyorum. Aynı satırlarda
    # aynı skoru vermiyorsa deneyin seçtiği güç ve eşik servise gitmiyor demektir;
    # bunu yorumla değil sayıyla kapatmak istedim.
    probe = np.unique(np.linspace(0, len(features)-1, 4000, dtype=int))
    served = ProductRiskContextEngine(policy).apply(features.iloc[probe].reset_index(drop=True),
                                                   scores.iloc[probe].reset_index(drop=True)).adjusted_anomaly_score
    difference = float(np.abs(served.to_numpy()-adjusted.iloc[probe].to_numpy()).max())
    equivalence = {"rows_checked": int(len(probe)), "max_absolute_difference": difference,
                   "engine": "fraud_case.product_context.ProductRiskContextEngine",
                   "gate": "Product risk applies only above training_end_transaction_dt, in both the experiment and the service.",
                   "threshold_source": "Train rows scored the way the service scores them, so the train quantile and the "
                                       "served score come from one function."}
    assert difference < 1e-12, equivalence
    summary = {"plan": plan, "global_train_rate": global_rate, "profiles": profiles, "candidates": candidates,
               "selected": selected, "later_validation_sensitivity": sensitivity, "train_derived_thresholds": same_threshold,
               "original_raw_threshold": old["threshold"], "serving_policy_changed": False,
               "training_end_transaction_dt": training_end, "serving_equivalence": equivalence}
    (folder / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    (folder / "policy.json").write_text(json.dumps(policy, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame({"TransactionID": source.loc[v_index, "TransactionID"], "adjusted_anomaly_score": adjusted.loc[v_index]}).to_parquet(folder / "validation_scores.parquet", index=False)
    lines = ["# İşlem tipi risk profili ile context deneyi", "",
             "Case'in 6. adımında önerilen işlem tipi bazlı risk profili uygulanır. Dört anomali katmanı ve eşit ağırlıklı "
             "raw skor değişmez. Önceki ürün grubunda alışıldık tutar indirimi, geçmiş etiketlerden öğrenilen ürün riski ile "
             "değiştirilir; diğer davranış ve takvim/güven kuralları korunur.", "",
             "Risk profili yalnızca train etiketlerinden hesaplanır. 1.000 sanal gözlemle genel fraud oranına yumuşatma ve "
             "en az 1.000 gerçek gözlem desteği kullanılır. Log risk oranı [-1,1] ile sınırlanır; skor düzeltmesi en fazla "
             "seçilen güç kadardır. Güçlü anomali koruması negatif düzeltmeyi bloke eder. Tanınmayan ürün düzeltme almaz. "
             "Skor yine kalibre edilmiş fraud olasılığı değildir.", "",
             "Train fraud etiketlerinin validation başlamadan bilindiği varsayılır; veri fraud bildirim gecikmesi sağlamaz. "
             "Bu nedenle context katmanı gözetimlidir. Önceki final test görülmüş olduğundan sonraki validation bir "
             "geliştirme kontrolüdür, bağımsız başarı kanıtı değildir. Test etiketleri bu deneyde okunmaz.", "",
             "| Ürün | Train işlem | Train fraud | Yumuşatılmış oran | Sınırlı log risk oranı |", "|---|---:|---:|---:|---:|"]
    for key, row in profiles.items():
        lines.append(f"| {key} | {row['count']} | {row['fraud_count']} | {row['smoothed_rate']:.3%} | {row['log_relative_risk']:.4f} |")
    lines += ["", "## Güç seçimi", "", "Erken validation iki kronolojik blokta değerlendirilir; aynı %5 inceleme bütçesinde en düşük blok TP kazanımı, sonra toplam TP ve AP kullanılır. Dört güç adayı etiketler okunmadan planda kaydedilir.", "",
              "| Güç | Erken validation TP | En düşük blok TP kazanımı | AP |", "|---|---:|---:|---:|"]
    for row in candidates:
        lines.append(f"| {row['strength']} | {row['calibration']['tp']} | {row['minimum_block_tp_gain']} | {row['calibration']['average_precision']:.4f} |")
    lines += ["", f"Seçilen güç: **{selected['strength']}**.", "", "## Sonraki validation: aynı inceleme bütçesi", "",
              "| Bütçe | Yöntem | İnceleme | TP | FP | Precision | Recall |", "|---|---|---:|---:|---:|---:|---:|"]
    for rate, group in sensitivity.items():
        for name, m in group.items():
            lines.append(f"| {float(rate):.0%} | {name} | {m['alerts']} | {m['tp']} | {m['fp']} | {m['precision']:.3%} | {m['recall']:.3%} |")
    lines += ["", f"AP raw: {sensitivity['0.05']['raw']['average_precision']:.4f}; ürün context: {sensitivity['0.05']['product_context']['average_precision']:.4f}.", "",
              "## Train'den seçilen eşiklerle alarm yükü", "", "Bu eşikler yaklaşık train üst %5'inden ayrı ayrı seçilir. Validation'da eşit alarm bütçesi garantilemez; yukarıdaki sabit bütçe kıyasıyla karıştırılmamalıdır.", "",
              f"Ürün riski yalnızca train dönemi bittikten sonra, yani TransactionDT > {training_end:.0f} olan işlemlerde "
              "uygulanır. Servis bu kapıyı zaten uyguluyordu, deney uygulamıyordu: eşik train satırlarına ürün "
              "düzeltmesi eklenerek hesaplanıyordu, yani servisin o satırlarda hiç üretmediği bir skor dağılımından. "
              "Deneyi servise eşitledim. Sözle bırakmamak için seçilen politikayı servisin kendi motoruna verip "
              f"{equivalence['rows_checked']} satırda karşılaştırdım; kesimin iki tarafını da kapsıyor ve en büyük "
              f"fark {difference:g}.", "",
              "Düzeltmenin görünür sonucu: train üst %5'i artık davranış skorundan hesaplandığı için ürün context'inin "
              "eşiği ham skorun eşiğiyle aynı çıkıyor. Yani aşağıdaki satırlar aynı eşikte karşılaştırma. Ürün riski "
              "skorları yukarı taşıdığı için eşiği geçen işlem sayısı artıyor; bu bir iyileşme değil, daha büyük bir "
              "inceleme yükü. Sabit bütçeli karşılaştırma yukarıdaki tabloda.", "",
              "| Yöntem | Eşik | Alarm | TP | FP |", "|---|---:|---:|---:|---:|"]
    for name, threshold in (("raw", old["threshold"]), ("product_context", selected["threshold"])):
        m = same_threshold[name]
        lines.append(f"| {name} | {threshold:.6f} | {m['alerts']} | {m['tp']} | {m['fp']} |")
    lines += ["", "Bu adayın raporu özgün final değerlendirmesinin yerine geçmez. Kural motoru ayrı değerlendirilmelidir; daha iyi skor sıralaması her sabit iş kuralının kararını otomatik iyileştirmez.", "",
              "Yeniden çalıştırma: `python scripts/evaluate_product_context.py`. Parametre ve çıktı: `artifacts/product_context/`.", ""]
    (ROOT / "reports/product_context.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"selected_strength": selected["strength"], "calibration": selected["calibration"],
                      "later_validation_5_percent": sensitivity["0.05"], "threshold_results": same_threshold}, indent=2), flush=True)


if __name__ == "__main__":
    run()
