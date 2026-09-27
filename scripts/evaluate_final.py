"""Tek seferlik final test. Burada hiçbir parametre fit edilmez, seçilmez.

`--report-only` sonuç dosyasına dokunmadan yalnızca rapor metnini kayıtlı
JSON'dan yeniden üretir: etiket okumaz, metrik hesaplamaz. Rapor metnini elle
düzeltmek yerine bunu ekledim, çünkü elle düzeltilen bir rapor artık
artifact'tan üretilmiş olmaz.
"""

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from fraud_case.evaluate_rules import decision_metrics
from fraud_case.evaluation import metrics

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-only", action="store_true",
                        help="Kayitli artifacts/final/evaluation.json dosyasindan yalnizca raporu yeniden uretir.")
    if parser.parse_args().report_only:
        out = ROOT / "artifacts/final"
        result = json.loads((out / "evaluation.json").read_text())
        render_report(result)
        print("Regenerated reports/final_evaluation.md from the stored result; no labels were read.", flush=True)
        return
    out = ROOT / "artifacts/final"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "evaluation.json").exists():
        raise RuntimeError("Final evaluation already exists; retain the original result instead of silently rerunning")
    freeze = {}
    paths = [*sorted((ROOT / "config").glob("*.json")), *sorted((ROOT / "src/fraud_case").glob("*.py")),
             ROOT / "artifacts/scoring/model.joblib", ROOT / "artifacts/context/selected_config.json",
             ROOT / "artifacts/context/product_reference.json", ROOT / "data/processed/scores.parquet",
             ROOT / "data/processed/context_scores.parquet", ROOT / "data/processed/rule_decisions.parquet"]
    for path in paths:
        with path.open("rb") as stream:
            freeze[str(path.relative_to(ROOT))] = hashlib.file_digest(stream, "sha256").hexdigest()
    # Bu snapshot'ı test etiketlerini AÇMADAN ÖNCE yaz
    (out / "frozen_manifest.json").write_text(json.dumps(freeze, indent=2))
    context = pd.read_parquet(ROOT / "data/processed/context_scores.parquet", filters=[("split", "=", "test")])
    rules = pd.read_parquet(ROOT / "data/processed/rule_decisions.parquet", filters=[("split", "=", "test")])
    labels = pd.read_parquet(ROOT / "data/processed/transactions.parquet", columns=["TransactionID", "isFraud"], filters=[("split", "=", "test")])
    if not labels.TransactionID.equals(context.TransactionID) or not labels.TransactionID.equals(rules.TransactionID):
        raise ValueError("Final test artifacts are not aligned")
    selection = json.loads((ROOT / "artifacts/context/evaluation.json").read_text())
    threshold = selection["threshold"]
    y = labels.isFraud
    raw = metrics(y, context.raw_anomaly_score, threshold=threshold)
    adjusted = metrics(y, context.adjusted_anomaly_score, threshold=threshold)
    budget = raw["alerts"]
    same_budget = {name: metrics(y, context[column], review_count=budget, transaction_ids=context.TransactionID)
                   for name, column in (("raw", "raw_anomaly_score"), ("context", "adjusted_anomaly_score"))}
    result = {"rows": len(labels), "threshold_from_train": threshold, "raw": raw, "context": adjusted,
              "same_review_budget": same_budget,
              "rules": decision_metrics(y, rules.rule_action.eq("review").to_numpy()),
              "context_at_rule_threshold_0_85": decision_metrics(y, context.adjusted_anomaly_score.ge(.85).to_numpy()),
              "frozen_before_labels": True, "parameter_changes_after_result": False,
              "history_policy": "Chronological unlabelled feature replay; scoring model and references remain train-fitted."}
    (out / "evaluation.json").write_text(json.dumps(result, indent=2, allow_nan=False))
    render_report(result)
    print(json.dumps(result, indent=2), flush=True)


def render_report(result):
    (ROOT / "reports/final_evaluation.md").write_text(final_report_lines(result), encoding="utf-8")


def final_report_lines(result):
    """Raporu yalnizca kayitli sonuctan uretir; etiket veya parquet okumaz."""
    raw, adjusted = result["raw"], result["context"]
    threshold, same_budget = result["threshold_from_train"], result["same_review_budget"]
    budget = raw["alerts"]
    lines = ["# Final test değerlendirmesi", "",
             f"Son kronolojik bölümde {result['rows']:,} işlem. Model, context ve rule yapılandırması ile kaynak kod hash'leri test etiketleri okunmadan önce kaydedildi. Bu sonuçlara göre parametre ayarı yapılmadı.", "",
             f"Train'den sabitlenen eşik: {threshold:.8f}. Validation'da seçilen context gücü değişmedi. Geçmiş özellikleri kronolojik ve etiketsiz güncellenir; model ve referanslar train'de sabittir.", "",
             "| Politika | İnceleme | TP | FP | Precision | Recall | FPR | AP |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "n/a" if value is None else f"%{100 * value:.3f}"
    for name, metric in (("Raw", raw), ("Context", adjusted), ("Rules", result["rules"])):
        lines.append(f"| {name} | {metric.get('alerts', metric.get('reviews')):,} | {metric['tp']:,} | {metric['fp']:,} | {percent(metric['precision'])} | {percent(metric['recall'])} | {percent(metric['false_positive_rate'])} | {percent(metric.get('average_precision'))} |")
    lines += ["", f"Aynı {budget:,} inceleme kapasitesinde raw {same_budget['raw']['tp']:,}, context {same_budget['context']['tp']:,} fraud yakaladı.", "",
              "Rules kararları skor sıralaması değildir ve kendi 0.85 yüksek-skor kuralı dahil farklı inceleme bütçesi "
              "kullanır; tablodan doğrudan ranking üstünlüğü çıkarılmaz. AP yalnızca sürekli raw/context skorlarına "
              "uygulanmıştır. Gerçek fraud oranı ve TP/FP sayıları, küçük mutlak recall farklarıyla birlikte "
              "değerlendirilmelidir.", "",
              "## Bu tablo hangi politikaya ait", "",
              "Üç satır da bu test koşulduğu andaki politikayı ölçüyor: davranış context'i seçilmiş 0.5 gücüyle açık. "
              "**Context** satırı doğrudan o skoru, **Rules** satırı da o skorlardan üretilen kural kararlarını "
              "gösteriyor; R09 eşiği `adjusted_anomaly_score` üzerinde çalıştığı için Rules satırı context'ten "
              "bağımsız değil.", "",
              "Servisin şu andaki varsayılanı `raw`, yani context indirimi uygulanmıyor. Dolayısıyla **Rules** satırı "
              "güncel varsayılanın final dönemindeki performansı değildir. Bunu ölçmek final testi yeniden açmak "
              "olurdu; açmadım. Ölçebildiğim şey aynı değişimin geliştirme verisindeki etkisiydi: iki profil sonraki "
              "validation yarısında 165 işlemde farklı karar veriyor ([varsayılan politika raporu]"
              "(default_policy.md)). Final dönemi için bu farkı bilmiyorum ve tahmin etmiyorum.", "",
              "**Raw** satırı context'ten etkilenmez; güncel varsayılanın skor tarafı bu satırdır.", "",
              "## Yorum", "",
              "Bu çalışma gözetimsiz anomali sıralaması ve açıklanabilir iş kurallarını birleştirir. Fraud etiketiyle "
              "optimize edilmiş bir sınıflandırıcı veya üretime hazır fraud performansı iddiası taşımaz. Daha fazla inceleme "
              "ile daha fazla fraud bulmak tek başına verimlilik artışı değildir. Bir sonraki model revizyonu yeni bir "
              "değerlendirme planı gerektirir; bu test artık görülmüştür.", "",
              "Kaynak snapshot: artifacts/final/frozen_manifest.json. Tüm metrikler: artifacts/final/evaluation.json. "
              "Bu metin `python scripts/evaluate_final.py --report-only` ile aynı JSON'dan yeniden üretilir; sonuç "
              "dosyası değişmez.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
