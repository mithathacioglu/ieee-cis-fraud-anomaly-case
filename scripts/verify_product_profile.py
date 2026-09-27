"""Çalışan ürün context'ini bağımsız geliştirme hesabıyla karşılaştırır."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from fraud_case.api import create_app
from fraud_case.evaluate_context import validation_sections
from fraud_case.evaluate_rules import decision_metrics
from fraud_case.product_context import ProductRiskContextEngine
from fraud_case.rules import RuleEngine

ROOT = Path(__file__).resolve().parents[1]


def main():
    filters = [("split", "=", "validation")]
    features = pd.read_parquet(ROOT / "data/processed/features.parquet", filters=filters).reset_index(drop=True)
    raw = pd.read_parquet(ROOT / "data/processed/scores.parquet", filters=filters).reset_index(drop=True)
    source = pd.read_parquet(ROOT / "data/processed/transactions.parquet", columns=["TransactionID", "TransactionDT", "ProductCD", "isFraud"], filters=filters).reset_index(drop=True)
    features["TransactionDT"], features["ProductCD"] = source.TransactionDT, source.ProductCD
    assert features.TransactionID.equals(raw.TransactionID) and features.TransactionID.equals(source.TransactionID)
    context = ProductRiskContextEngine.from_json(ROOT / "artifacts/product_context/policy.json")
    scored = context.apply(features, raw)
    independent = pd.read_parquet(ROOT / "artifacts/product_context/validation_scores.parquet")
    assert independent.TransactionID.equals(scored.TransactionID)
    np.testing.assert_allclose(scored.adjusted_anomaly_score, independent.adjusted_anomaly_score, rtol=1e-12, atol=1e-12)
    parts = scored[[f"context_{layer}_contribution" for layer in ("column", "multivariate", "entity", "temporal")]].sum(axis=1)
    np.testing.assert_allclose(parts+scored.product_context_adjustment, scored.adjusted_anomaly_score, rtol=1e-12, atol=1e-12)
    combined = pd.concat([features, scored.drop(columns=list(set(features.columns) & set(scored.columns)))], axis=1)
    engine = RuleEngine.from_json(ROOT / "config/rules.json")
    decisions = engine.evaluate(combined)
    _, audit, _ = validation_sections(features, .5)
    current_rules = pd.read_parquet(ROOT / "data/processed/rule_decisions.parquet", filters=filters).reset_index(drop=True)
    assert current_rules.TransactionID.equals(features.TransactionID)
    effect = {"baseline": decision_metrics(source.loc[audit, "isFraud"], current_rules.loc[audit, "rule_action"].eq("review").to_numpy()),
              "product_risk": decision_metrics(source.loc[audit, "isFraud"], decisions.loc[audit, "rule_action"].eq("review").to_numpy())}
    indices = np.linspace(0, len(features)-1, 64, dtype=int)
    with TestClient(create_app(ROOT, context_profile="product_risk")) as client:
        for index in indices:
            response = client.post("/explain", json={"transaction_id": int(features.iloc[index].TransactionID), "include_rag": False})
            response.raise_for_status()
            body = response.json()
            assert body["context_profile"] == "product_risk"
            assert np.isclose(body["scores"]["adjusted_anomaly_score"], scored.iloc[index].adjusted_anomaly_score, atol=1e-12)
            assert np.isclose(body["scores"]["raw_anomaly_score"], raw.iloc[index].raw_anomaly_score, atol=1e-12)
            assert body["rule_decision"] == decisions.iloc[index].rule_action
            assert np.isclose(body["explanation"]["context"]["adjusted_score"], body["scores"]["adjusted_anomaly_score"], atol=1e-12)
    files = ["src/fraud_case/product_context.py", "src/fraud_case/service.py", "src/fraud_case/api.py",
             "src/fraud_case/container.py", "artifacts/product_context/policy.json"]
    result = {"validation_rows_verified": len(features), "api_requests_verified": len(indices),
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
              "later_validation_rules": effect, "test_labels_read": False}
    (ROOT / "artifacts/platform/product_profile_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Ürün riski profilinin servis doğrulaması", "",
             "118.108 validation işleminin yeni context skorları bağımsız deney betiğiyle karşılaştırıldı. Davranış katman "
             "katkılarının toplamı + ürün riski düzeltmesi = final adjusted skor özdeşliği bütün satırlarda doğrulandı. 64 "
             "işlemde gerçek model kullanan API skoru, açıklaması ve kural kararı batch çıktısıyla eşleşti. Bu kontrol yeni "
             "final test değildir.", "",
             "## Aynı 10 kuralın sonraki validation etkisi", "", "| Profil | İnceleme | TP | FP | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]
    for name, m in effect.items():
        lines.append(f"| {name} | {m['reviews']} | {m['tp']} | {m['fp']} | {m['precision']:.3%} | {m['recall']:.3%} |")
    lines += ["", "Kuralların sayısı ve eşikleri değiştirilmedi; R09 hâlâ adjusted>=0.85 kullanır. Yeni context skor dağılımı bu kuralın alarm sayısını değiştirebilir. Bu tablo farklı inceleme bütçeleri içerdiğinden, doğrudan sıralama üstünlüğü anlamına gelmez. Skorun sabit bütçeli karşılaştırması [ürün context raporundadır](product_context.md).", "",
              "Yeniden çalıştırma: `python scripts/verify_product_profile.py`. Gerçek HTTP ve LLM kontrolü ayrıca `python "
              "scripts/verify_http.py --context-profile product_risk` ile yapılır.", ""]
    (ROOT / "reports/product_profile_verification.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
