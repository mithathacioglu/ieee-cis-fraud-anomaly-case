import json
import shutil
from dataclasses import asdict
from pathlib import Path

import httpx
import joblib
import pandas as pd
import pytest
from dependency_injector import providers
from fastapi.testclient import TestClient

from fraud_case.aggregation import ScoreAggregator
from fraud_case.anomaly import AnomalyEngine
from fraud_case.api import create_app
from fraud_case.container import Container
from fraud_case.context import ContextConfig
from fraud_case.features import FeatureBuilder
from fraud_case.rag import GroundingError, LocalModelError, OllamaClient, RagConfig


@pytest.fixture(scope="module")
def prepared(tmp_path_factory):
    root = tmp_path_factory.mktemp("api_artifacts")
    for folder in ("config", "artifacts/scoring", "artifacts/context", "data/processed"):
        (root / folder).mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parents[1]
    for name in ("rules.json", "rag.json"):
        shutil.copyfile(source / "config" / name, root / "config" / name)
    rows = [{"TransactionID": i + 1, "TransactionDT": float(i * 180), "TransactionAmt": float(100 + (i % 4) * 10),
             "card1": 1., "card2": 2., "card3": 3., "card5": 5., "addr1": 6., "card4": "visa",
             "ProductCD": "W", "P_emaildomain": "mail.example", "R_emaildomain": "mail.example",
             "DeviceInfo": "example-device", "has_identity": True} for i in range(40)]
    data = pd.DataFrame(rows)
    features = FeatureBuilder().transform(data)
    features["split"] = "train"
    engine = AnomalyEngine(n_estimators=8, max_samples=32).fit(features)
    aggregator = ScoreAggregator().fit(engine.score(features))
    joblib.dump({"engine": engine, "aggregator": aggregator}, root / "artifacts/scoring/model.joblib")
    (root / "artifacts/context/selected_config.json").write_text(json.dumps(asdict(ContextConfig())))
    (root / "artifacts/context/product_reference.json").write_text("{}")
    features.to_parquet(root / "data/processed/features.parquet", index=False)
    # Bu fixture'da etiket kolonu yok
    data.to_parquet(root / "data/processed/transactions.parquet", index=False)
    return root, rows


@pytest.fixture
def client(prepared):
    root, _ = prepared
    with TestClient(create_app(root)) as client:
        yield client


def test_score_uses_real_model_and_finite_json(client):
    response = client.post("/score", json={"transaction_id": 25})
    assert response.status_code == 200
    result = response.json()
    assert 0 <= result["scores"]["raw_anomaly_score"] <= 1
    assert result["mode"] == "saved_causal_features"
    assert len(result["agent_trace"]) == 2
    assert result["agent_trace"][1]["sender"] == "scoring_agent"
    assert "NaN" not in response.text and "Infinity" not in response.text


def test_rules_and_explain_use_the_same_decision(client):
    payload = {"transaction_id": 25}
    score = client.post("/score", json=payload).json()
    rules = client.post("/rules/evaluate", json=payload).json()
    detail = client.post("/explain", json={**payload, "include_rag": False}).json()
    assert score["rule_decision"] == rules["action"] == detail["rule_decision"]
    assert detail["explanation"]["anomaly"]["observed_features"]["prior_transaction_count"] == 24
    assert score["scores"]["raw_anomaly_score"] == detail["scores"]["raw_anomaly_score"]


def test_raw_history_mode_matches_saved_features(client, prepared):
    _, rows = prepared
    saved = client.post("/score", json={"transaction_id": 25}).json()
    response = client.post("/score", json={"transaction": rows[24], "history": list(reversed(rows[:24]))})
    assert response.status_code == 200
    raw = response.json()
    assert raw["mode"] == "request_local_history"
    assert raw["scores"]["raw_anomaly_score"] == pytest.approx(saved["scores"]["raw_anomaly_score"])


def test_repeated_and_other_requests_do_not_mutate_history(client, prepared):
    _, rows = prepared
    body = {"transaction": rows[24], "history": rows[:24]}
    first = client.post("/score", json=body).json()["scores"]
    client.post("/score", json={"transaction": {**rows[30], "TransactionAmt": 100000.}})
    assert client.post("/score", json=body).json()["scores"] == first


@pytest.mark.parametrize("payload", [{}, {"transaction_id": "25"}, {"transaction_id": True},
                                      {"transaction_id": 25, "isFraud": 1},
                                      {"transaction_id": 25, "calendar_reference": "2020-01-01"}])
def test_invalid_request_returns_422(client, payload):
    assert client.post("/score", json=payload).status_code == 422


def test_unknown_id_and_future_history(client, prepared):
    assert client.post("/score", json={"transaction_id": 999999}).status_code == 404
    _, rows = prepared
    for history in ([rows[25]], [rows[24]], [{**rows[23], "TransactionDT": rows[24]["TransactionDT"]}]):
        response = client.post("/score", json={"transaction": rows[24], "history": history})
        assert response.status_code == 422


def test_missing_artifacts_returns_503(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        assert client.post("/score", json={"transaction_id": 1}).status_code == 503


class StubRag:
    """Test double only: no fake generation path exists in the application."""
    def retrieve(self, question):
        return [{"id": "POL-PRIORITY", "text": "Example"}]
    def reason(self, question, sources, evidence=None):
        assert evidence["decision"] in {"review", "monitor", "no_rule_match"}
        return {"answer": "Test explanation", "citations": ["POL-PRIORITY"], "abstained": False}
    def query(self, question):
        return {"answer": "Test response", "citations": ["POL-PRIORITY"], "abstained": False}


def test_four_agents_delegate_with_same_request_id(prepared):
    root, _ = prepared
    container = Container(root=root)
    with container.rag.override(providers.Object(StubRag())):
        with TestClient(create_app(root, container)) as client:
            response = client.post("/explain", json={"transaction_id": 25})
            assert response.status_code == 200
            body = response.json()
            trace = body["agent_trace"]
            assert [e["recipient"] for e in trace] == ["scoring_agent", "rule_agent", "knowledge_agent", "reasoning_agent"]
            assert {e["request_id"] for e in trace} == {body["request_id"]}
            assert all(e["status"] == "completed" for e in trace)
            assert client.post("/rag/query", json={"question": "Priority?"}).status_code == 200


@pytest.mark.parametrize("error,status", [(LocalModelError("offline model unavailable"), 503), (GroundingError("bad citation"), 502)])
def test_rag_failure_is_explicit_not_a_fake_answer(prepared, error, status):
    root, _ = prepared
    class BrokenRag(StubRag):
        def query(self, question):
            raise error
    container = Container(root=root)
    with container.rag.override(providers.Object(BrokenRag())):
        with TestClient(create_app(root, container)) as client:
            assert client.post("/rag/query", json={"question": "Why?"}).status_code == status
            assert client.post("/score", json={"transaction_id": 25}).status_code == 200


def test_api_schema_contains_required_endpoints(client):
    paths = client.get("/openapi.json").json()["paths"]
    assert {"/score", "/explain", "/rules/evaluate", "/rag/query"} <= set(paths)
    assert client.get("/health").json()["artifacts"]["scoring"]


@pytest.mark.parametrize("extra", [{"answer": "The transaction is safe with 85 percent fraud probability."},
                                  {"decision": "approved"}, {"adjusted_anomaly_score": 0.0}])
def test_generated_claims_cannot_override_api_evidence(prepared, extra):
    root, _ = prepared
    selection = {"source_ids": ["POL-SCORES"], "language": "tr", "abstained": False, **extra}
    local = OllamaClient(RagConfig(), transport=httpx.MockTransport(lambda request: httpx.Response(
        200, json={"message": {"content": json.dumps(selection)}})))
    class SelectingRag:
        def retrieve(self, question):
            return [{"id": "POL-SCORES", "text": "The score is not a fraud probability."}]
        def reason(self, question, sources, evidence=None):
            return local.answer(question, sources, evidence)
    container = Container(root=root)
    try:
        with container.rag.override(providers.Object(SelectingRag())):
            with TestClient(create_app(root, container)) as client:
                before = client.post("/score", json={"transaction_id": 25}).json()
                assert client.post("/explain", json={"transaction_id": 25}).status_code == 502
                after = client.post("/score", json={"transaction_id": 25}).json()
                assert before["scores"] == after["scores"] and before["rule_decision"] == after["rule_decision"]
    finally:
        local.close()


def test_external_trust_is_request_scoped_and_same_time_is_not_prior(client, prepared):
    _, rows = prepared
    timestamp = rows[24]["TransactionDT"]
    base = {"transaction_id": 25, "include_rag": False}
    baseline = client.post("/explain", json=base).json()
    evidence = {"trusted_entity": True, "trust_observed_at": timestamp - 1,
                "trust_source": "synthetic-test-review"}
    response = client.post("/explain", json={**base, "external_context": evidence})
    assert response.status_code == 200
    body = response.json()
    assert body["explanation"]["context"]["trust_available"]
    assert body["scores"]["context_trust_available"]
    assert body["scores"]["raw_anomaly_score"] == baseline["scores"]["raw_anomaly_score"]
    # Geçerli güven kaydı olsa bile yüksek velocity indirimi bloke etmeli
    assert body["scores"]["context_guard_active"]
    assert body["scores"]["context_reduction"] == 0
    for observed_at in (timestamp, timestamp + 1):
        result = client.post("/explain", json={**base, "external_context": {**evidence, "trust_observed_at": observed_at}})
        assert result.status_code == 200
        assert not result.json()["scores"]["context_trust_available"]
    assert client.post("/explain", json=base).json()["scores"] == baseline["scores"]


@pytest.mark.parametrize("external", [
    {"trusted_entity": True},
    {"trusted_entity": True, "trust_source": " ", "trust_observed_at": 1.0},
    {"weekend_activity_expected": True, "schedule_source": "schedule"},
    {"trusted_entity": "true"},
    {"trust_observed_at": -1.0},
    {"TransactionID": 26},
])
def test_invalid_external_context_is_rejected(client, external):
    assert client.post("/score", json={"transaction_id": 25, "external_context": external}).status_code == 422


def test_weekend_schedule_reaches_scoring_and_rule_endpoints(client, prepared):
    _, rows = prepared
    payload = {"transaction": rows[24], "history": rows[:24],
               "calendar_reference": "2026-09-26T00:00:00+03:00",
               "external_context": {"weekend_activity_expected": True,
                                    "schedule_observed_at": 0.0, "schedule_source": "synthetic-schedule"}}
    response = client.post("/explain", json={**payload, "include_rag": False})
    assert response.status_code == 200
    body = response.json()
    assert body["scores"]["context_calendar_available"]
    assert body["scores"]["context_weekend_schedule_available"]
    assert body["explanation"]["context"]["observed_context"]["is_weekend"] == 1
    rules = client.post("/rules/evaluate", json=payload)
    assert rules.status_code == 200
    assert rules.json()["action"] == body["rule_decision"]


def test_default_profile_serves_raw_scores(prepared):
    root, _ = prepared
    stored = json.loads((root / "artifacts/context/selected_config.json").read_text())
    with TestClient(create_app(root)) as plain, TestClient(create_app(root, context_profile="behavior_context")) as adjusted:
        assert plain.get("/health").json()["context_profile"] == "raw"
        assert adjusted.get("/health").json()["context_profile"] == "behavior_context"
        # Rapor "context'i açmamak gerekir" diyor. O cümle kodda da doğru olmalı:
        # varsayılan profil indirim gücünü sıfırlıyor, istenirse açıkça açılıyor.
        assert plain.app.state.container.scoring().context.config.strength == 0.
        assert adjusted.app.state.container.scoring().context.config.strength == stored["strength"] > 0
        for transaction_id in (10, 25, 40):
            body = plain.post("/score", json={"transaction_id": transaction_id}).json()
            assert body["context_profile"] == "raw"
            assert body["scores"]["context_reduction"] == 0
            assert body["scores"]["adjusted_anomaly_score"] == body["scores"]["raw_anomaly_score"]


def test_missing_optional_product_policy_is_explicit_and_profile_is_reported(prepared):
    root, _ = prepared
    with TestClient(create_app(root, context_profile="product_risk")) as client:
        assert client.get("/health").json()["context_profile"] == "product_risk"
        assert not client.get("/health").json()["artifacts"]["product_context"]
        assert client.post("/score", json={"transaction_id": 25}).status_code == 503
    with pytest.raises(ValueError, match="Unknown context profile"):
        create_app(root, context_profile="made_up")
