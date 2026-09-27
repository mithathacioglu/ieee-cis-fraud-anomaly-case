"""Dışarıdan verilen context'i API üzerinden, kayıtlı model dosyalarıyla dener.

Aşağıdaki takvim ve güven senaryoları sentetik gösterim; IEEE-CIS verisi veya
fraud başarısı ölçümü değil.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from fraud_case.api import create_app

ROOT = Path(__file__).resolve().parents[1]


def main():
    base = dict(card1=1., card2=2., card3=3., card5=5., addr1=6., card4="visa", ProductCD="W",
                P_emaildomain="example.com", R_emaildomain="example.com", DeviceInfo="demo-device", has_identity=True)
    history = [dict(base, TransactionID=900000+i, TransactionDT=float(i*14400),
                    TransactionAmt=float(100+(i % 3-1)*5)) for i in range(30)]
    transaction = dict(base, TransactionID=900100, TransactionDT=432000., TransactionAmt=100.)
    payload = {"transaction": transaction, "history": history, "include_rag": False}
    trust = dict(trusted_entity=True, trust_observed_at=1., trust_source="synthetic-demo-review")
    schedule = dict(weekend_activity_expected=True, schedule_observed_at=1., schedule_source="synthetic-demo-schedule")
    scenarios = [
        ("unknown_calendar", {}, set()),
        ("weekday_business_hours", {"calendar_reference": "2026-09-23T10:00:00+03:00"}, {"business_hours"}),
        ("weekend_without_schedule", {"calendar_reference": "2026-09-21T10:00:00+03:00"}, set()),
        ("weekend_with_schedule", {"calendar_reference": "2026-09-21T10:00:00+03:00", "external_context": schedule}, {"expected_weekend"}),
        ("prior_trust", {"external_context": trust}, {"verified_trust"}),
        ("same_second_trust", {"external_context": {**trust, "trust_observed_at": 432000.}}, set()),
    ]
    results = []
    # Senaryolar ve 64 satirlik tekrar behavior_context ile kosuyor: kayitli
    # context_scores.parquet o profille uretildi. Varsayilan profil ("raw")
    # ayrica asagida kontrol ediliyor.
    with TestClient(create_app(ROOT, context_profile="behavior_context")) as client:
        for name, extra, expected in scenarios:
            response = client.post("/explain", json={**payload, **extra})
            response.raise_for_status()
            body = response.json()
            context = body["explanation"]["context"]
            assert set(context["matched_rules"]) == expected, (name, context)
            assert {item["rule"] for item in context["applied"]} == expected
            assert (context["reduction"] > 0) == bool(expected)
            results.append({"scenario": name, "status": response.status_code, "raw_score": context["raw_score"],
                            "adjusted_score": context["adjusted_score"], "reduction": context["reduction"],
                            "applied": context["applied"]})
        assert len({item["raw_score"] for item in results}) == 1
        # Varsayılan skorlar değişiklik öncesi kayıtlı çıktıyı hâlâ üretmeli
        frame = pd.read_parquet(ROOT / "data/processed/context_scores.parquet")
        sample = frame.iloc[np.linspace(0, len(frame)-1, 64, dtype=int)]
        fields = ["raw_anomaly_score", "adjusted_anomaly_score", "context_reduction"]
        fields += [f"{layer}_{suffix}" for layer in ("column", "multivariate", "entity", "temporal")
                   for suffix in ("raw", "normalized", "contribution")]
        for row in sample.to_dict("records"):
            response = client.post("/score", json={"transaction_id": int(row["TransactionID"])})
            response.raise_for_status()
            scores = response.json()["scores"]
            for field in fields:
                actual = np.nan if scores[field] is None else scores[field]
                assert np.isclose(actual, row[field], rtol=1e-10, atol=1e-12, equal_nan=True), (row["TransactionID"], field)
        schema = client.get("/openapi.json").json()
    # Varsayilan profil context indirimi UYGULAMAMALI. Rapor "context'i acmadim"
    # diyorsa bunu iddia olarak birakmak yetmez; burada olcuyorum.
    default_scenarios = []
    with TestClient(create_app(ROOT)) as client:
        assert client.get("/health").json()["context_profile"] == "raw"
        for name, extra, expected in scenarios:
            response = client.post("/explain", json={**payload, **extra})
            response.raise_for_status()
            context = response.json()["explanation"]["context"]
            # Kural eslesmeleri hala raporlaniyor; degisen tek sey indirim.
            assert set(context["matched_rules"]) == expected, (name, context)
            assert context["applied"] == [] and context["reduction"] == 0.
            assert context["adjusted_score"] == context["raw_score"]
            default_scenarios.append({"scenario": name, "matched_rules": sorted(context["matched_rules"]),
                                      "reduction": context["reduction"]})
        for row in sample.to_dict("records"):
            response = client.post("/score", json={"transaction_id": int(row["TransactionID"])})
            response.raise_for_status()
            scores = response.json()["scores"]
            assert np.isclose(scores["raw_anomaly_score"], row["raw_anomaly_score"], rtol=1e-10, atol=1e-12)
            assert scores["adjusted_anomaly_score"] == scores["raw_anomaly_score"]
            assert scores["context_reduction"] == 0.
    evidence = {"scope": "Synthetic API scenarios using frozen real scoring artifacts; no fraud labels read.",
                "behavior_profile_score_replays": len(sample), "default_score_replays": len(sample),
                "default_score_fields": fields, "scenarios": results,
                "default_profile": "raw",
                "default_profile_applies_no_reduction": True,
                "default_profile_scenarios": default_scenarios,
                "default_profile_note": "Development showed no fixed-budget gain from the behavioural context, so the "
                                        "serving default applies no reduction. Rule matches are still reported.",
                "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ("src/fraud_case/api.py", "src/fraud_case/service.py")}}
    folder = ROOT / "artifacts/platform"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "context_api_verification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    (folder / "openapi.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    lines = ["# API bağlam senaryoları", "", "Bu örnekler sentetiktir. Gerçek IEEE-CIS takvimi veya güven kaydı değildir; fraud başarısı ölçmez.",
             "Kaydedilmiş gerçek anomali modeli ve seçilmiş context ayarları değiştirilmeden API üzerinden çalıştırılmıştır.", "",
             "| Senaryo | Raw | Adjusted | Uygulanan bağlam |", "|---|---:|---:|---|"]
    for item in results:
        rules = ", ".join(rule["rule"] for rule in item["applied"]) or "yok"
        lines.append(f"| {item['scenario']} | {item['raw_score']:.6f} | {item['adjusted_score']:.6f} | {rules} |")
    lines += ["", "Altı senaryoda raw skor aynıdır. Yalnızca geçerli bağlam ilgili katkıyı azaltır; aynı saniyede alınan güven kaydı geçmiş bilgi sayılmaz.",
              "64 kayıtlı işlemde API'nin 15 skor alanı, değişiklik öncesi kaydedilen sonuçlarla tolerans içinde eşleşmiştir.", "",
              "Bu tekrarın kapsamını olduğundan geniş göstermemek gerekir: kayıtlı feature'lardan ve kaydedilmiş "
              "modelden başlıyor, yani skorlama, birleştirme ve context katmanını doğruluyor. Feature üretimini "
              "yeniden türetmiyor; dolayısıyla geçmiş feature'larının nasıl hesaplandığındaki bir değişikliği tek "
              "başına yakalamaz. O katman `scripts/verify_features.py` ile ayrı ve farklı bir yöntemle kontrol "
              "ediliyor, ayrıca dondurulmuş parquet hash'leri sonucun değişmediğini gösteriyor.", "",
              "## Varsayılan profil", "",
              "Yukarıdaki tablo `behavior_context` profiliyle üretildi; `data/processed/context_scores.parquet` de o profile ait. "
              "Servisin varsayılanı `raw`: context indirimi uygulanmaz. Bunu iddia olarak bırakmıyorum, aynı koşuda ölçüyorum. "
              "Varsayılan profille altı senaryonun tamamında eşleşen kural listesi aynı kalır, uygulanan indirim boş ve "
              "adjusted skor raw skora eşittir; 64 kayıtlı işlemde de `context_reduction` sıfırdır.", "",
              "| Senaryo | Eşleşen kural | İndirim |", "|---|---|---:|",
              *[f"| {item['scenario']} | {', '.join(item['matched_rules']) or 'yok'} | {item['reduction']:.6f} |" for item in default_scenarios],
              "", "Kural eşleşmesini kapatmadım, yalnızca skora etkisini kapattım: inceleme ekibi hangi bağlamın tuttuğunu "
              "yine görür, karar skoru bundan etkilenmez.", "",
              "Yeniden çalıştırma: `python scripts/verify_context_api.py`. Bu doğrulama fraud etiketi okumaz.", ""]
    (ROOT / "reports/context_api.md").write_text("\n".join(lines), encoding="utf-8")
    print("Verified 6 synthetic context API scenarios, 64 frozen-score replays and a default profile that applies no reduction.", flush=True)


if __name__ == "__main__":
    main()
