"""Gerçek API çalıştırmalarından çevrimdışı bir what-if gezgini kurar."""

import hashlib
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

from fastapi.testclient import TestClient

from fraud_case.api import create_app

ROOT = Path(__file__).resolve().parents[1]
AMOUNTS = (100, 1000)
PACES = ("regular", "burst")
TRUST = ("none", "prior", "same_time")
CALENDARS = ("unknown", "weekday", "weekend", "weekend_planned")


def scenario_request(amount, pace, trust, calendar):
    base = dict(card1=1., card2=2., card3=3., card5=5., addr1=6., card4="visa", ProductCD="W",
                P_emaildomain="example.com", R_emaildomain="example.com", DeviceInfo="demo-device", has_identity=True)
    history = [dict(base, TransactionID=900000+i, TransactionDT=float(i*14400),
                    TransactionAmt=float(100+(i % 3-1)*5)) for i in range(30)]
    if pace == "burst":
        for i in range(18, 30):
            history[i]["TransactionDT"] = float(432000-(30-i)*60)
    body = {"transaction": dict(base, TransactionID=900100, TransactionDT=432000., TransactionAmt=float(amount)),
            "history": history, "include_rag": False}
    if calendar != "unknown":
        # Sentetik takvim: hedef işlem pazartesi veya cumartesi 10:00'a düşüyor
        body["calendar_reference"] = "2026-09-23T10:00:00+03:00" if calendar == "weekday" else "2026-09-21T10:00:00+03:00"
    external = {}
    if trust != "none":
        external.update(trusted_entity=True, trust_observed_at=1. if trust == "prior" else 432000.,
                        trust_source="synthetic-demo-review")
    if calendar == "weekend_planned":
        external.update(weekend_activity_expected=True, schedule_observed_at=1., schedule_source="synthetic-demo-schedule")
    if external:
        body["external_context"] = external
    return body


def main():
    states, audit = {}, []
    with TestClient(create_app(ROOT, context_profile="behavior_context")) as client:
        for amount, pace, trust, calendar in product(AMOUNTS, PACES, TRUST, CALENDARS):
            request = scenario_request(amount, pace, trust, calendar)
            response = client.post("/explain", json=request)
            response.raise_for_status()
            body = response.json()
            key = f"{amount}|{pace}|{trust}|{calendar}"
            rules = body["explanation"]["rules"]
            state = {"scores": body["scores"], "context": body["explanation"]["context"],
                     "features": {**body["explanation"]["context"]["observed_context"],
                                  **body["explanation"]["anomaly"]["observed_features"]},
                     "decision": body["rule_decision"], "winner": body["winning_rule"],
                     "rules": [r for r in rules["rules"] if r["matched"]], "conflict": rules["action_conflict"],
                     "agent_trace": body["agent_trace"], "limitations": body["limitations"]}
            states[key] = state
            parts = sum(body["scores"][f"context_{layer}_contribution"] for layer in ("column", "multivariate", "entity", "temporal"))
            assert abs(parts-body["scores"]["adjusted_anomaly_score"]) < 1e-12
            audit.append({"key": key, "request": request, "response": body})
    assert len(states) == 48
    # Context sadece final skoru değil, hiçbir raw katmanı değiştirmemeli
    for amount, pace in product(AMOUNTS, PACES):
        base = states[f"{amount}|{pace}|none|unknown"]["scores"]
        for trust, calendar in product(TRUST, CALENDARS):
            actual = states[f"{amount}|{pace}|{trust}|{calendar}"]["scores"]
            for field in ("raw_anomaly_score", "column_raw", "multivariate_raw", "entity_raw", "temporal_raw"):
                assert actual[field] == base[field], (amount, pace, field)
    plain = states["100|regular|none|unknown"]
    trusted = states["100|regular|prior|unknown"]
    same_time = states["100|regular|same_time|unknown"]
    burst = states["100|burst|prior|unknown"]
    assert trusted["scores"]["adjusted_anomaly_score"] < plain["scores"]["adjusted_anomaly_score"]
    assert same_time["scores"]["adjusted_anomaly_score"] == plain["scores"]["adjusted_anomaly_score"]
    assert not same_time["context"]["trust_available"]
    assert burst["context"]["guard_active"] and burst["context"]["trust_available"]
    assert burst["context"]["reduction"] == 0 and burst["decision"] == "review"
    assert burst["winner"] == "R02_hourly_velocity"
    assert states["100|regular|none|weekend_planned"]["context"]["reduction"] > states["100|regular|none|weekend"]["context"]["reduction"]
    assert states["1000|regular|prior|unknown"]["context"]["guard_active"]
    evaluation_path = ROOT / "artifacts/product_context/evaluation.json"
    measured = json.loads(evaluation_path.read_text(encoding="utf-8"))["later_validation_sensitivity"]
    for budget in measured.values():
        assert budget["raw"]["alerts"] == budget["product_context"]["alerts"]
        for key in ("raw", "product_context"):
            assert budget[key]["tp"] + budget[key]["fp"] == budget[key]["alerts"]
    recorded = json.loads((ROOT / "artifacts/demo/full/demo.json").read_text(encoding="utf-8"))
    for name, expected in recorded["source_sha256"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == expected, name
    model_examples = [{"id": c["transaction_id"], "title": c["title"],
                       "trace": c["response"]["agent_trace"], "rag": c["response"]["rag"]} for c in recorded["cases"]]
    payload = {"created_utc": datetime.now(timezone.utc).isoformat(), "states": states,
               "measured": measured, "model_examples": model_examples,
               "scope": "48 synthetic requests scored by frozen real API; cached results, not live browser inference. Performance panel uses existing development aggregates."}
    template = (ROOT / "scripts/scenario_demo.html").read_text(encoding="utf-8")
    encoded = json.dumps(payload, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    output = ROOT / "reports/scenario_demo.html"
    output.write_text(template.replace("__DATA__", encoded), encoding="utf-8")
    folder = ROOT / "artifacts/demo/scenarios"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "requests_and_responses.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    proof = {"scenarios_verified": len(states), "raw_layers_unchanged_by_context": True,
             "context_contributions_sum_to_adjusted_score": True,
             "prior_trust_reduces_stable_score": True, "same_time_trust_ignored": True,
             "burst_guard_blocks_trust_discount": True, "weekend_schedule_required": True,
             "existing_llm_examples_reused": len(model_examples), "fraud_labels_read": False,
             "evaluation_sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
             "source_sha256": recorded["source_sha256"], "html_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}
    (folder / "verification.json").write_text(json.dumps(proof, indent=2), encoding="utf-8")
    (ROOT / "reports/scenario_demo.md").write_text("\n".join([
        "# İşlem senaryoları ve bütçe karşılaştırması", "", "Aç: `reports/scenario_demo.html`. Yeniden üret: `python scripts/scenario_demo.py`.", "",
        "48 sentetik istek, gerçek kayıtlı model ve baseline context ile `/explain` üzerinden hesaplandı. Tarayıcı bu "
        "kayıtlı sonuçlar arasında geçiş yapar; canlı çıkarım yapmaz. Girdi ve çıktılar "
        "`artifacts/demo/scenarios/requests_and_responses.json` içinde yereldir.", "",
        "## İki dakikalık anlatım", "", "1. Güven indirimi: işlem ve geçmiş aynıyken, işlemden önce alınan güven kaydının sınırlı skor etkisini göster.",
        "2. Yoğunluk koruması: aynı güven kaydıyla son saatte 12 geçmiş işleme geç; R02 inceleme önerir ve indirim bloke olur.",
        "3. Geç gelen güven: işlemle aynı anda alınan kaydın indirim sağlayamadığını göster.",
        "4. Hafta sonu: açık takvim tek başına yetmez; önceden bilinen faaliyet programı eklenince uygun senaryoda indirim uygulanır.",
        "5. Bütçe: gerçek geliştirme sonuçlarında yüzde 5 kazanımı ile yüzde 1 kaybını karşılaştır; aynı inceleme sayısını koru.",
        "6. Yerel LLM: kaydedilmiş gerçek üç işlemde dört ajan izini ve Qwen3 kaynak seçimini aç.", "",
        "Takvim, güven ve işlem geçmişi senaryoları sentetiktir; gerçek IEEE-CIS müşteri bilgisi değildir. Bütçe paneli "
        "sentetik senaryolardan üretilmez: mevcut 59.054 satırlık sonraki validation sonuçlarını okur. Bu veri daha "
        "önce görüldüğü için bağımsız test değildir. Kontroller LLM çağırmaz; LLM paneli önceki gerçek yerel çağrıların "
        "kaydıdır.", "",
        "Model, ağırlıklar, eşikler ve kurallar bu gösterim için değiştirilmedi. 48 senaryoda bağlamın raw katmanları "
        "değiştirmediği; güven zaman sınırı, yoğunluk koruması ve hafta sonu programı şartı doğrulandı.", "",
    ]), encoding="utf-8")
    print(json.dumps({"html": str(output), "scenarios": len(states),
                      "stable_before": plain["scores"]["adjusted_anomaly_score"],
                      "stable_trusted": trusted["scores"]["adjusted_anomaly_score"],
                      "trusted_burst_decision": burst["decision"], "trusted_burst_rule": burst["winner"]}), flush=True)


if __name__ == "__main__":
    main()
