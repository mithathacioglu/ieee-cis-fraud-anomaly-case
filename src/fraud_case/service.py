"""Skorlama servisi. Ya kayıtlı bir transaction id, ya işlem + kendi geçmişi."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd

from fraud_case.context import ContextConfig, ContextEngine
from fraud_case.features import REQUIRED_COLUMNS, FeatureBuilder
from fraud_case.product_context import ProductRiskContextEngine


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


class FeatureRepository:
    """Read-only snapshot: lazy load once; no labels loaded by the serving process."""

    def __init__(self, root: Path):
        self.root, self._frame, self._lock = root, None, Lock()

    def get(self, transaction_id):
        with self._lock:
            if self._frame is None:
                features = pd.read_parquet(self.root / "data/processed/features.parquet")
                source = pd.read_parquet(self.root / "data/processed/transactions.parquet",
                                         columns=["TransactionID", "TransactionDT", "ProductCD"])
                if not features.TransactionID.equals(source.TransactionID) or features.TransactionID.duplicated().any():
                    raise ValueError("Feature repository ID alignment failed")
                features["TransactionDT"] = source.TransactionDT
                features["ProductCD"] = source.ProductCD
                self._frame = features.set_index("TransactionID", drop=False)
            if transaction_id not in self._frame.index:
                raise KeyError(transaction_id)
            return self._frame.loc[[transaction_id]].reset_index(drop=True).copy()


# Servisin uygulayabilecegi context politikalari.
#
# Varsayilan "raw": context indirimi uygulanmaz. Sabit inceleme butcesinde
# context ek fraud yakalamadi, sadece daha az alarm uretti; basa bas maliyet
# orani 6.8 (reports/context_adjustment.md). Kazandirmayan bir duzeltmeyi
# varsayilan yapmazdim, o yuzden burada da varsayilan degil. Acmak cagiranin
# acik tercihi olmali.
#
# "raw" profili context motorunu strength=0 ile kurar: hangi kuralin esleser
# oldugu /explain icinde gorunmeye devam eder, skora dokunmaz. Boylece "kapali"
# demek aciklamayi da kapatmak anlamina gelmiyor.
CONTEXT_PROFILES = ("raw", "behavior_context", "product_risk")


@dataclass
class ScoringResult:
    features: pd.DataFrame
    scores: pd.DataFrame
    mode: str
    limitations: list[str]
    anomaly_explanation: dict
    context_explanation: dict
    context_profile: str = "raw"

    def public(self):
        return json_safe({"transaction_id": int(self.features.TransactionID.iloc[0]), "mode": self.mode, "context_profile": self.context_profile,
                          "scores": self.scores.iloc[0].to_dict(), "limitations": self.limitations})


class ScoringService:
    def __init__(self, root: Path, repository: FeatureRepository, context_profile="raw"):
        self.root, self.repository = root, repository
        if context_profile not in CONTEXT_PROFILES:
            raise ValueError("Unknown context profile")
        self.context_profile = context_profile
        # Sadece bu projenin kendi ürettiği model dosyası yüklensin
        model = joblib.load(root / "artifacts/scoring/model.joblib")
        self.engine, self.aggregator = model["engine"], model["aggregator"]
        folder = root / "artifacts/context"
        config = ContextConfig(**json.loads((folder / "selected_config.json").read_text()))
        if context_profile == "raw":
            # Ayni yapilandirma, gucu sifirlanmis: butun indirimler contribution * rate
            # * strength oldugu icin skor raw ile birebir ayni kaliyor.
            config = replace(config, strength=0.)
        self.context = ContextEngine(config, json.loads((folder / "product_reference.json").read_text()))
        if context_profile == "product_risk":
            self.context = ProductRiskContextEngine.from_json(root / "artifacts/product_context/policy.json")

    def score(self, transaction_id=None, transaction=None, history=None, calendar_reference=None, external_context=None):
        if (transaction_id is None) == (transaction is None):
            raise ValueError("Choose exactly one saved transaction ID or raw transaction")
        if transaction_id is not None:
            if history or calendar_reference:
                raise ValueError("Saved transaction lookup cannot override history or calendar")
            features = self.repository.get(transaction_id)
            mode = "saved_causal_features"
            limitations = ["Entity is a card/address proxy; score is not a fraud probability."]
        else:
            rows = [*list(history or []), transaction]
            frame = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
            if frame.TransactionID.isna().any() or frame.TransactionID.duplicated().any():
                raise ValueError("History and current transaction IDs must be unique")
            if not frame.iloc[:-1].TransactionDT.lt(transaction["TransactionDT"]).all():
                raise ValueError("Supplied history must be strictly earlier than the target")
            frame = frame.sort_values(["TransactionDT", "TransactionID"], kind="stable").reset_index(drop=True)
            built = FeatureBuilder(calendar_reference).transform(frame)
            features = built.loc[built.TransactionID.eq(transaction["TransactionID"])].reset_index(drop=True)
            features["TransactionDT"] = float(transaction["TransactionDT"])
            features["ProductCD"] = transaction.get("ProductCD")
            mode = "request_local_history"
            limitations = ["History and global category frequencies cover only the supplied request, not the full production stream.",
                           "Calendar reference, if provided, is caller-supplied and not verified by IEEE-CIS.",
                           "Entity is a card/address proxy; score is not a fraud probability."]
        raw = self.aggregator.transform(self.engine.score(features))
        raw.insert(0, "TransactionID", features.TransactionID)
        external = None
        if external_context is not None:
            if "TransactionID" in external_context:
                raise ValueError("External context is bound to the requested transaction")
            external = pd.DataFrame([{**external_context, "TransactionID": int(features.TransactionID.iloc[0])}])
            limitations.append("External trust/schedule is caller-supplied; source identity is not independently authenticated.")
        if self.context_profile == "behavior_context":
            limitations.append("Behavioral context showed no fixed-budget gain in development and is not the default profile; "
                               "see reports/context_adjustment.md for the break-even cost ratio.")
        if self.context_profile == "product_risk":
            limitations.append("Product risk uses historical training fraud labels assumed available before this transaction; development "
                               "results are not an independent test.")
        adjusted = self.context.apply(features, raw, external)
        return ScoringResult(features, adjusted, mode, limitations,
                             self.engine.explain(features), self.context.explain(features, raw, external), self.context_profile)

    @staticmethod
    def rule_inputs(result):
        return pd.concat([result.features, result.scores.drop(columns="TransactionID")], axis=1)
