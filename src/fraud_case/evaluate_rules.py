"""Kuralları bütün işlemlerde çalıştırır, kararları sonraki validation'da ölçer."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_case.rules import RuleEngine


def decision_metrics(labels, decisions):
    y, flagged = np.asarray(labels), np.asarray(decisions)
    if y.ndim != 1 or y.shape != flagged.shape or not len(y) or not np.isin(y, [0, 1]).all() or flagged.dtype != bool:
        raise ValueError("Expected nonempty aligned binary labels and boolean decisions")
    positive = y.astype(bool)
    tp, fp = int((positive & flagged).sum()), int((~positive & flagged).sum())
    fn, tn = int((positive & ~flagged).sum()), int((~positive & ~flagged).sum())
    return {"rows": len(y), "reviews": tp + fp, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": tp / (tp + fn) if tp + fn else None,
            "false_positive_rate": fp / (fp + tn) if fp + tn else None}


def load_inputs(root):
    features = pd.read_parquet(root / "data/processed/features.parquet")
    scores = pd.read_parquet(root / "data/processed/context_scores.parquet")
    if not features.index.equals(scores.index) or not features.TransactionID.equals(scores.TransactionID):
        raise ValueError("Rule feature and score rows must have identical ID order and index")
    overlap = set(features.columns) & set(scores.columns)
    for column in overlap:
        if not features[column].equals(scores[column]):
            raise ValueError(f"Conflicting rule input column: {column}")
    return pd.concat([features, scores.drop(columns=sorted(overlap))], axis=1)


def run(root: Path, rules_path: Path):
    engine = RuleEngine.from_json(rules_path)
    frame = load_inputs(root)
    result = engine.evaluate(frame)
    result.insert(1, "split", frame.split)
    assert result.TransactionID.equals(frame.TransactionID)
    assert result.loc[result.matched_rule_count.eq(0), "rule_action"].eq("no_rule_match").all()
    assert result.loc[result.matched_rule_count.gt(0), "winning_rule"].notna().all()

    context_eval = json.loads((root / "artifacts/context/evaluation.json").read_text())
    boundary = context_eval["validation_boundary_seconds"]
    # Sadece geliştirmede zaten görülmüş sonraki validation bölümü açılıyor
    labels = pd.read_parquet(root / "data/processed/transactions.parquet", columns=["TransactionID", "isFraud"],
                             filters=[("split", "=", "validation"), ("TransactionDT", ">=", boundary)])
    indexed = labels.set_index("TransactionID")
    audit_index = frame.index[frame.TransactionID.isin(indexed.index)]
    audit = frame.loc[audit_index]
    if not audit.split.eq("validation").all() or len(audit) != len(labels):
        raise ValueError("Evaluation label IDs do not align with validation inputs")
    y = indexed.loc[audit.TransactionID, "isFraud"].to_numpy()
    decisions = result.loc[audit_index]
    # Sabit karşılaştırma eşiği; optimize edilmiş işletme eşiği değil
    threshold = .85
    comparison = {
        "raw_at_0_85": decision_metrics(y, audit.raw_anomaly_score.ge(threshold).to_numpy()),
        "context_at_0_85": decision_metrics(y, audit.adjusted_anomaly_score.ge(threshold).to_numpy()),
        "rule_review": decision_metrics(y, decisions.rule_action.eq("review").to_numpy()),
    }
    activity = []
    for rule in engine.config["rules"]:
        rule_id = rule["id"]
        activity.append({"id": rule_id, "priority": rule["priority"], "action": rule["then"]["action"],
                         "enabled": rule["enabled"], "description": rule["description"], "condition": rule["when"],
                         "all_matches": int(result[f"rule_match_{rule_id}"].sum()),
                         "audit_matches": int(decisions[f"rule_match_{rule_id}"].sum()),
                         "audit_wins": int(decisions.winning_rule.eq(rule_id).sum())})
    examples = {}
    for name, mask in {
        "action_conflict": decisions.rule_action_conflict,
        "multiple_matches": decisions.matched_rule_count.gt(1),
        "review": decisions.rule_action.eq("review"),
        "monitor": decisions.rule_action.eq("monitor"),
        "no_rule_match": decisions.rule_action.eq("no_rule_match"),
    }.items():
        if mask.any():
            examples[name] = engine.explain(frame.loc[[mask.index[mask][0]]])
    summary = {
        "config_sha256": engine.config_sha256, "rows": len(frame), "rules": activity,
        "evaluation_segment": "Later validation development segment; already used during context development, not a blind holdout.",
        "validation_boundary_seconds": boundary, "test_labels_used": False,
        "comparison_threshold": threshold, "audit_metrics": comparison,
        "audit_action_conflicts": int(decisions.rule_action_conflict.sum()),
        "audit_multiple_matches": int(decisions.matched_rule_count.gt(1).sum()),
        "actions_by_split": {str(split): group.rule_action.value_counts().to_dict() for split, group in result.groupby("split")},
        "score_policy": "Rules route transactions; raw and adjusted scores remain unchanged. No rule-driven ranking is defined.",
        "threshold_policy": "Illustrative business hypotheses fixed before this rules evaluation; no rule tuning using these labels.",
    }
    files = {"rules_json": rules_path, **{name: root / name for name in (
        "data/processed/features.parquet", "data/processed/context_scores.parquet",
        "data/processed/transactions.parquet", "artifacts/context/evaluation.json")}}
    for name, path in files.items():
        with path.open("rb") as stream:
            summary.setdefault("input_sha256", {})[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    output = root / "artifacts/rules"
    output.mkdir(parents=True, exist_ok=True)
    for filename, value in (("evaluation.json", summary), ("rules_snapshot.json", engine.config), ("examples.json", examples)):
        (output / filename).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    result.to_parquet(root / "data/processed/rule_decisions.parquet", index=False)
    render_report(root, summary, examples)
    print(json.dumps({"rows": len(frame), "rules": len(activity), "audit_metrics": comparison,
                      "audit_action_conflicts": summary["audit_action_conflicts"], "example_types": list(examples),
                      "test_labels_used": False}, indent=2), flush=True)


def render_report(root, summary, examples):
    lines = ["# Rule engine", "",
             "JSON kuralları feature ve anomali skorlarından inceleme kararı üretir. Raw/context skorları değişmez. Kurallar "
             "fraud etiketi üretmez; `review` inceleme önerisi, `monitor` izleme önerisidir. `no_rule_match` güvenli/izin "
             "verilmiş işlem anlamına gelmez.", "",
             "## Kural sözleşmesi", "",
             "`config/rules.json`: version, default_action ve rules. Her kural id, enabled, priority, description, when ve "
             "then içerir. `all` / `any` iç içe kullanılabilir. Operatörler: eq, ne, gt, gte, lt, lte, in, is_missing, "
             "is_present. Yalnızca kayıtlı sayısal/boolean feature ve skor adları kullanılabilir; etiket, ID, Python ifadesi "
             "ve bilinmeyen alanlar reddedilir.", "",
             "Yüksek sayısal priority kazanır. Eşitlikte alfabetik kural ID'si kazanır; JSON sırası sonucu değiştirmez. "
             "Bütün eşleşmeler ve kaybeden kurallar açıklamada tutulur. Review kuralları mevcut dosyada monitor kuralından "
             "yüksek önceliklidir; yeni bir yapılandırmada önceliklerin sorumluluğu politika sahibindedir.", "",
             "Eksik hücre normal karşılaştırmalarda false olur; NaN != eşik de eşleşmez. Eksikliği aramak için is_missing "
             "açıkça seçilir. Eksik kolon, hatalı yapılandırma sayılır ve hata verir. Devre dışı kuralın girdisi aranmaz. "
             "Koşullar arasında kısa devreyle açıklama atlanmaz; bütün koşulların gözlenen değeri raporlanır.", "",
             "Motor her başlatmada JSON'u doğrular ve değişmez bir kopyasını alır. Dosyayı değiştirdikten sonra yeni "
             "RuleEngine.from_json(...) örneği yeni politikayı çalıştırır; mevcut istek ortasında politika değişmez. "
             "Otomatik dosya izleyici yoktur. Yapılandırma özeti her açıklamada bulunur.", "",
             "## Kurallar ve çalışma kapsamı", "",
             "Eşikler örnek iş politikası hipotezleridir; kurum tarafından verilmiş veya fraud başarısına göre optimize "
             "edilmiş eşikler değildir. Ülke veya gerçek yerel gece bilgisi olmadığı için bu tür kurallar uydurulmadı. "
             "Kart/adres entity'si gerçek kişi, DeviceInfo da benzersiz cihaz kimliği değildir. Rare-pair kurallarındaki "
             "1000 geçmiş işlem şartı global başlangıç filtresidir; her kategori çifti için 1000 gözlem anlamına gelmez.", "",
             "| Kural | Öncelik | Karar | Tüm veride eşleşme | Validation eşleşme | Validation kazanan |",
             "|---|---:|---|---:|---:|---:|"]
    for rule in summary["rules"]:
        lines.append(f"| {rule['id']} | {rule['priority']} | {rule['action']} | {rule['all_matches']:,} | {rule['audit_matches']:,} | {rule['audit_wins']:,} |")
    lines += ["", "Koşulların tam değerleri JSON'dadır. Bir işlem birden fazla kuralla eşleşebilir; eşleşme sayıları toplanarak işlem sayısı bulunamaz.", "",
              "## Geliştirme verisinde karar etkisi", "",
              "Context geliştirmesinde görülmüş sonraki validation bölümü kullanıldı; bağımsız final holdout değildir. Kural "
              "eşikleri bu çalıştırmanın etiketli sonuçlarına bakılarak değiştirilmedi. Final test etiketleri okunmadı. "
              "Raw/context karşılaştırma eşiği sabit 0.85'tir; önceki context raporunun train quantile eşiğiyle birebir aynı "
              "değildir.", "",
              "Buradaki `rule_review` satırı `data/processed/context_scores.parquet` üzerinden üretildi, yani davranış "
              "context'i uygulanmış skorlara ait. R09 eşiği `adjusted_anomaly_score` alanında çalıştığı için servisin "
              "varsayılan `raw` profili aynı kararları vermiyor; aynı JSON kurallarını çalıştırmak aynı kararı üretmek "
              "anlamına gelmez. İki profilin farkı [varsayılan politika raporunda](default_policy.md) ölçüldü.", "",
              "| Politika | İnceleme | TP | FP | FN | Precision | Recall | FPR |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    def percent(value):
        return "n/a" if value is None else f"%{100 * value:.3f}"
    for name, metric in summary["audit_metrics"].items():
        lines.append(f"| {name} | {metric['reviews']:,} | {metric['tp']:,} | {metric['fp']:,} | {metric['fn']:,} | {percent(metric['precision'])} | {percent(metric['recall'])} | {percent(metric['false_positive_rate'])} |")
    lines += ["", "Bu politikalar eşit sayıda işlem incelemeye göndermez. Daha çok fraud yakalama, daha yüksek inceleme yükünden kaynaklanabilir; tek başına model/sıralama iyileşmesi kanıtı değildir. Rule engine bir sıralama skoru tanımlamaz; bu nedenle karar etiketlerinden yapay AP/ROC üretilmez. Operasyon kapasitesi ve fraud maliyeti verilmeden üretimde hangi politikanın kullanılacağına karar verilemez.", "",
              f"Validation bölümünde {summary['audit_multiple_matches']:,} işlem birden fazla kuralla eşleşti; {summary['audit_action_conflicts']:,} işlemde review/monitor çakışması çözüldü.", "",
              "## Gerçek işlem açıklamaları", ""]
    for name, explanation in examples.items():
        matched = [r["id"] for r in explanation["rules"] if r["matched"]]
        lines.append(f"- {name}: TransactionID {explanation['transaction_id']}; eşleşmeler {', '.join(matched) or 'yok'}; kazanan {explanation['winning_rule'] or 'yok'}; karar {explanation['action']}.")
    lines += ["", "Her açıklamada koşul ağacı, eşik, gözlenen değer, eksiklik, selected/superseded/not_matched/disabled durumu ve yapılandırma SHA-256 bulunur. Ayrıntılar `artifacts/rules/examples.json` içindedir.", "",
              "## Yeniden üretim", "", "```powershell", ".\\.venv\\Scripts\\python.exe -m fraud_case.evaluate_rules", "```", "",
              "Özel politika: `--rules config/my_rules.json`. Yeniden çalıştırma yerel rules çıktılarının üstüne yazar. Son "
              "kullanılan politikanın kopyası `artifacts/rules/rules_snapshot.json` içinde saklanır.", "",
              "- `data/processed/rule_decisions.parquet`: ID/split, kural eşleşmeleri, kazanan, karar ve çakışma bayrağı.",
              "- `artifacts/rules/evaluation.json`: politika özeti, kaynak dosya hash'leri, sayımlar ve karar metrikleri.",
              "- `tests/test_rules.py`: öncelik, çakışma, eşitlik, eksik veri, doğrulama, bütün kuralların erişilebilirliği "
              "ve açıklama testleri.", ""]
    (root / "reports/rule_engine.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--rules", type=Path, default=Path("config/rules.json"))
    args = parser.parse_args()
    root = args.root.resolve()
    run(root, args.rules if args.rules.is_absolute() else root / args.rules)


if __name__ == "__main__":
    main()
