"""Skorlama, açıklama, iş kuralları ve yerel RAG için FastAPI endpoint'leri."""

import os
from contextlib import asynccontextmanager, suppress
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fraud_case.container import Container
from fraud_case.rag import GroundingError, LocalModelError
from fraud_case.service import CONTEXT_PROFILES, json_safe


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Transaction(StrictModel):
    TransactionID: int = Field(ge=0)
    TransactionDT: float = Field(ge=0)
    TransactionAmt: float = Field(ge=0)
    card1: float | None = None
    card2: float | None = None
    card3: float | None = None
    card5: float | None = None
    addr1: float | None = None
    card4: str | None = Field(default=None, max_length=100)
    ProductCD: str | None = Field(default=None, max_length=100)
    P_emaildomain: str | None = Field(default=None, max_length=300)
    R_emaildomain: str | None = Field(default=None, max_length=300)
    DeviceInfo: str | None = Field(default=None, max_length=500)
    has_identity: bool = False


class ExternalContext(StrictModel):
    """Caller assertions, with provenance and times on the TransactionDT axis."""

    trusted_entity: bool = False
    trust_observed_at: float | None = Field(default=None, ge=0)
    trust_source: str | None = Field(default=None, min_length=1, max_length=300)
    weekend_activity_expected: bool = False
    schedule_observed_at: float | None = Field(default=None, ge=0)
    schedule_source: str | None = Field(default=None, min_length=1, max_length=300)

    @model_validator(mode="after")
    def evidence_required(self):
        for flag, timestamp, source in (
            (self.trusted_entity, self.trust_observed_at, self.trust_source),
            (self.weekend_activity_expected, self.schedule_observed_at, self.schedule_source),
        ):
            if flag and (timestamp is None or source is None or not source.strip()):
                raise ValueError("Enabled external context requires observation time and a nonblank source")
        return self


class ScoreRequest(StrictModel):
    transaction_id: int | None = Field(default=None, ge=0)
    transaction: Transaction | None = None
    history: list[Transaction] = Field(default_factory=list, max_length=1000)
    calendar_reference: str | None = Field(default=None, max_length=100)
    external_context: ExternalContext | None = None

    @model_validator(mode="after")
    def one_source(self):
        if (self.transaction_id is None) == (self.transaction is None):
            raise ValueError("Supply exactly one transaction_id or transaction")
        if self.transaction_id is not None and (self.history or self.calendar_reference):
            raise ValueError("Saved transaction cannot override history/calendar")
        return self

    def score_payload(self):
        return self.model_dump(include={"transaction_id", "transaction", "history", "calendar_reference", "external_context"})


class ExplainRequest(ScoreRequest):
    include_rag: bool = True
    question: str | None = Field(default=None, min_length=1, max_length=2000)


class RagRequest(StrictModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(root=None, container=None, context_profile=None):
    root = Path(root or os.environ.get("FRAUD_CASE_ROOT", Path.cwd())).resolve()
    profile = context_profile or os.environ.get("FRAUD_CONTEXT_PROFILE", "raw")
    if profile not in CONTEXT_PROFILES:
        raise ValueError("Unknown context profile")
    container = container or Container(root=root, context_profile=profile)

    @asynccontextmanager
    async def lifespan(app):
        yield
        with suppress(FileNotFoundError):
            container.client().close()

    app = FastAPI(title="Fraud Anomaly Case", version="0.1.0", lifespan=lifespan,
                  description="Local case-study API. Scores are anomaly ranks; decisions are investigation recommendations.")
    app.state.container = container

    @app.exception_handler(KeyError)
    async def not_found(request: Request, exc):
        return JSONResponse(status_code=404, content={"detail": "Transaction ID was not found."})

    @app.exception_handler(FileNotFoundError)
    async def unavailable(request: Request, exc):
        return JSONResponse(status_code=503, content={"detail": "Required local artifact is missing. Complete the preparation commands."})

    @app.exception_handler(LocalModelError)
    async def model_unavailable(request: Request, exc):
        return JSONResponse(status_code=503, content={"detail": str(exc)})

    @app.exception_handler(GroundingError)
    async def ungrounded(request: Request, exc):
        return JSONResponse(status_code=502, content={"detail": str(exc)})

    @app.exception_handler(ValueError)
    async def invalid(request: Request, exc):
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    @app.get("/health")
    def health():
        artifacts = {name: (root / path).exists() for name, path in {
            "scoring": "artifacts/scoring/model.joblib", "context": "artifacts/context/selected_config.json",
            "features": "data/processed/features.parquet", "rag_index": "artifacts/rag/index.json"}.items()}
        if container.context_profile() == "product_risk":
            artifacts["product_context"] = (root / "artifacts/product_context/policy.json").exists()
        return {"status": "ok", "artifacts": artifacts, "llm_checked": False, "context_profile": container.context_profile()}

    @app.post("/score")
    def score(body: ScoreRequest):
        result = container.coordinator().run(body.score_payload())
        result.pop("explanation")
        return json_safe(result)

    @app.post("/explain")
    def explain(body: ExplainRequest):
        coordinator = container.rag_coordinator() if body.include_rag else container.coordinator()
        return json_safe(coordinator.run(body.score_payload(), include_rag=body.include_rag, question=body.question))

    @app.post("/rules/evaluate")
    def evaluate_rules(body: ScoreRequest):
        result = container.coordinator().run(body.score_payload())
        return {"request_id": result["request_id"], "transaction_id": result["transaction_id"],
                **result["explanation"]["rules"], "agent_trace": result["agent_trace"]}

    @app.post("/rag/query")
    def query(body: RagRequest):
        return container.rag().query(body.question)

    return app


app = create_app()
