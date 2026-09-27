"""Feature üretimi.

Buradaki tek kritik kural: aynı saniyedeki satırların hepsi önce hesaplanır,
sonra geçmişe eklenir. Kaynak saniye veriyor, sıra vermiyor; yani aynı
saniyedeki iki işlem birbirinin geçmişi sayılamaz.
"""

import math
from collections import Counter, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from itertools import groupby

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = [
    "TransactionID", "TransactionDT", "TransactionAmt", "card1", "card2", "card3",
    "card5", "addr1", "card4", "ProductCD", "P_emaildomain", "R_emaildomain",
    "DeviceInfo", "has_identity",
]

# Feature sözleşmesi burası. ID, etiket ve split adı listede yok.
FEATURE_SPECS = {
    "amount": ("column", "Current transaction amount; original units."),
    "log_amount": ("column", "log(1 + current transaction amount)."),
    "relative_day": ("temporal", "Elapsed seconds // 86400; not a calendar day."),
    "hour_phase": ("temporal", "Hour within the undisclosed reference's 24-hour cycle."),
    "hour_sin": ("temporal", "Sine of the relative 24-hour phase."),
    "hour_cos": ("temporal", "Cosine of the relative 24-hour phase."),
    "hour": ("temporal", "Calendar hour; null unless an explicit reference is supplied."),
    "day_of_week": ("temporal", "Monday=0; null without an explicit reference."),
    "is_weekend": ("context", "Saturday/Sunday; null without an explicit reference."),
    "is_business_hours": ("context", "Weekday 09:00-17:00 in the configured fixed offset; otherwise null."),
    "calendar_available": ("context", "An explicit timezone-aware reference was supplied."),
    "has_identity": ("context", "Identity row exists; not evidence of trust."),
    "entity_known": ("entity", "All five card/address proxy components are present."),
    "prior_transaction_count": ("entity", "Number of strictly earlier transactions for the proxy."),
    "prior_mean_amount": ("entity", "Mean amount of strictly earlier proxy transactions."),
    "prior_std_amount": ("entity", "Population standard deviation of earlier amounts; null before two observations."),
    "amount_to_prior_mean": ("entity", "Current amount / prior mean; null when mean is zero or absent."),
    "amount_zscore": ("entity", "Signed deviation / prior std; requires five observations and positive std."),
    "prior_count_1h": ("temporal", "Proxy transactions in [t-3600, t)."),
    "prior_count_24h": ("temporal", "Proxy transactions in [t-86400, t)."),
    "seconds_since_previous": ("temporal", "Time since the most recent strictly earlier proxy transaction."),
    "history_span_days": ("entity", "Elapsed time since the proxy's first earlier observation."),
    "prior_distinct_products": ("relational", "Distinct observed products in proxy history."),
    "prior_distinct_email_domains": ("relational", "Distinct payer email domains in proxy history."),
    "prior_distinct_devices": ("relational", "Distinct observed device descriptions in proxy history."),
    "entity_product_is_new": ("relational", "Current product absent from nonempty observed product history; otherwise null."),
    "entity_email_is_new": ("relational", "Current payer domain absent from nonempty observed domain history; otherwise null."),
    "entity_device_is_new": ("relational", "Current device description absent from nonempty observed device history; otherwise null."),
    "entity_hour_probability": ("temporal", "Laplace-smoothed prior hour-phase frequency; null for no history."),
    "prior_global_count": ("relational", "Number of strictly earlier transactions across the stream."),
    "product_prior_frequency": ("relational", "Earlier product count / earlier rows with observed product."),
    "email_prior_frequency": ("relational", "Earlier payer domain count / earlier rows with observed payer domain."),
    "device_prior_frequency": ("relational", "Earlier device count / earlier rows with observed device description."),
    "product_card_prior_frequency": ("relational", "Earlier product/card-brand pair count / complete earlier pairs."),
    "email_pair_prior_frequency": ("relational", "Earlier payer/recipient domain pair count / complete earlier pairs."),
    "product_card_prior_support": ("relational", "Number of earlier observations supporting the current product/card pair."),
    "email_pair_prior_support": ("relational", "Number of earlier observations supporting the current email pair."),
    "email_domains_match": ("relational", "Payer and recipient email domains match; null if either is absent."),
    "history_sufficient": ("context", "Complete proxy with at least five earlier transactions."),
    "frequent_entity": ("context", "At least 20 earlier transactions and at least seven days of observed history; not a trust label."),
    "usual_amount": ("context", "At least five earlier observations and amount/prior mean between 2/3 and 1.5; otherwise null or false."),
}


def category(value):
    if pd.isna(value):
        return None
    if isinstance(value, (int, float, np.number)):
        number = float(value)
        return str(int(number)) if number.is_integer() else str(number)
    return str(value)


@dataclass
class EntityHistory:
    count: int = 0
    mean: float = 0.0
    m2: float = 0.0
    first_time: float = 0.0
    last_time: float = 0.0
    hour_events: deque = field(default_factory=deque)
    day_events: deque = field(default_factory=deque)
    products: set = field(default_factory=set)
    emails: set = field(default_factory=set)
    devices: set = field(default_factory=set)
    hours: Counter = field(default_factory=Counter)

    def advance(self, timestamp: float) -> None:
        while self.hour_events and self.hour_events[0] < timestamp - 3600:
            self.hour_events.popleft()
        while self.day_events and self.day_events[0] < timestamp - 86400:
            self.day_events.popleft()

    def observe(self, timestamp, amount, product, email, device):
        if not self.count:
            self.first_time = timestamp
        self.count += 1
        delta = amount - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (amount - self.mean)
        self.last_time = timestamp
        self.hour_events.append(timestamp)
        self.day_events.append(timestamp)
        self.hours[int(timestamp // 3600) % 24] += 1
        for observed, value in ((self.products, product), (self.emails, email), (self.devices, device)):
            if value is not None:
                observed.add(value)


class FeatureBuilder:
    """Single-writer stream state. A batch must contain complete timestamp groups.

    Earlier validation/test events may update unlabelled history in a chronological
    replay. This is an explicit online-history simulation, not a frozen-history test.
    """

    def __init__(self, calendar_reference: str | None = None):
        self.reference = datetime.fromisoformat(calendar_reference) if calendar_reference else None
        if self.reference is not None and self.reference.utcoffset() is None:
            raise ValueError("calendar_reference must include a UTC offset")
        self.entities: dict[tuple, EntityHistory] = {}
        self.counts = {name: Counter() for name in ("product", "email", "device", "product_card", "email_pair")}
        self.totals = Counter()
        self.global_count = 0
        self.last_timestamp: float | None = None

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        missing = set(REQUIRED_COLUMNS) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing feature inputs: {sorted(missing)}")
        if frame.empty:
            raise ValueError("Feature batch is empty")
        if frame.TransactionID.isna().any() or frame.TransactionID.duplicated().any():
            raise ValueError("TransactionID must be present and unique within a batch")
        for column in ("TransactionDT", "TransactionAmt"):
            if not pd.api.types.is_numeric_dtype(frame[column]) or not np.isfinite(frame[column]).all() or (frame[column] < 0).any():
                raise ValueError(f"Invalid {column}")
        if not frame.TransactionDT.is_monotonic_increasing:
            raise ValueError("Feature inputs must be chronological")
        if self.last_timestamp is not None and frame.TransactionDT.iloc[0] <= self.last_timestamp:
            raise ValueError("A new batch must start after the previous complete timestamp group")
        if not frame.has_identity.isin([True, False]).all():
            raise ValueError("has_identity must be boolean")
        if self.reference is not None:
            # Geçmişe dokunmadan önce sınır tarihini doğrula
            self.reference + timedelta(seconds=float(frame.TransactionDT.iloc[-1]))
        output = []
        rows = frame[REQUIRED_COLUMNS].itertuples(index=False, name=None)
        for timestamp, same_time in groupby(rows, key=lambda row: row[1]):
            pending = []
            for row in same_time:
                transaction_id, _, amount, *rest = row
                card1, card2, card3, card5, addr1, card4, product, email, recipient, device, has_identity = map(category, rest)
                parts = (card1, card2, card3, card5, addr1)
                key = parts if all(part is not None for part in parts) else None
                state = self.entities.get(key) if key is not None else None
                if state is not None:
                    state.advance(timestamp)
                product_card = (product, card4) if product is not None and card4 is not None else None
                email_pair = (email, recipient) if email is not None and recipient is not None else None
                relations = {"product": product, "email": email, "device": device,
                             "product_card": product_card, "email_pair": email_pair}
                features = self._features(timestamp, float(amount), key, state, relations, bool(row[-1]))
                output.append({"TransactionID": transaction_id, **features})
                pending.append((key, timestamp, float(amount), relations))
            # Aynı saniyedeki satırlar birbirini görmesin
            for key, timestamp, amount, relations in pending:
                if key is not None:
                    if key not in self.entities:
                        self.entities[key] = EntityHistory()
                    self.entities[key].observe(timestamp, amount, relations["product"], relations["email"], relations["device"])
                for name, value in relations.items():
                    if value is not None:
                        self.counts[name][value] += 1
                        self.totals[name] += 1
                self.global_count += 1
            self.last_timestamp = float(timestamp)
        return pd.DataFrame(output, columns=["TransactionID", *FEATURE_SPECS])

    def _features(self, timestamp, amount, key, state, relations, has_identity):
        nan = float("nan")
        count = state.count if state is not None else 0
        hour_phase = int(timestamp // 3600) % 24
        angle = 2 * math.pi * (timestamp % 86400) / 86400
        prior_mean = state.mean if count else nan
        prior_std = math.sqrt(max(state.m2 / count, 0)) if count >= 2 else nan
        ratio = amount / prior_mean if count and prior_mean > 0 else nan
        span = (timestamp - state.first_time) / 86400 if count else nan
        calendar = self.reference + timedelta(seconds=float(timestamp)) if self.reference is not None else None
        result = {
            "amount": amount, "log_amount": math.log1p(amount),
            "relative_day": int(timestamp // 86400), "hour_phase": hour_phase,
            "hour_sin": math.sin(angle), "hour_cos": math.cos(angle),
            "hour": calendar.hour if calendar else nan, "day_of_week": calendar.weekday() if calendar else nan,
            "is_weekend": int(calendar.weekday() >= 5) if calendar else nan,
            "is_business_hours": int(calendar.weekday() < 5 and 9 <= calendar.hour < 17) if calendar else nan,
            "calendar_available": int(calendar is not None), "has_identity": int(has_identity),
            "entity_known": int(key is not None), "prior_transaction_count": count if key is not None else nan,
            "prior_mean_amount": prior_mean, "prior_std_amount": prior_std, "amount_to_prior_mean": ratio,
            "amount_zscore": (amount - prior_mean) / prior_std if count >= 5 and prior_std > 0 else nan,
            "prior_count_1h": len(state.hour_events) if state is not None else (0 if key is not None else nan),
            "prior_count_24h": len(state.day_events) if state is not None else (0 if key is not None else nan),
            "seconds_since_previous": timestamp - state.last_time if count else nan, "history_span_days": span,
            "prior_distinct_products": len(state.products) if state is not None else (0 if key is not None else nan),
            "prior_distinct_email_domains": len(state.emails) if state is not None else (0 if key is not None else nan),
            "prior_distinct_devices": len(state.devices) if state is not None else (0 if key is not None else nan),
            "entity_hour_probability": (state.hours[hour_phase] + 1) / (count + 24) if count else nan,
            "prior_global_count": self.global_count,
            "email_domains_match": int(relations["email_pair"][0] == relations["email_pair"][1]) if relations["email_pair"] else nan,
            "history_sufficient": int(count >= 5), "frequent_entity": int(count >= 20 and span >= 7),
            "usual_amount": int(2 / 3 <= ratio <= 1.5) if count >= 5 and np.isfinite(ratio) else nan,
        }
        for name, attribute in (("product", "products"), ("email", "emails"), ("device", "devices")):
            history = getattr(state, attribute) if state is not None else set()
            value = relations[name]
            result[f"entity_{name}_is_new"] = int(value not in history) if history and value is not None else nan
        for name, value in relations.items():
            support = self.counts[name][value] if value is not None else nan
            result[f"{name}_prior_frequency"] = support / self.totals[name] if self.totals[name] and value is not None else nan
            if name in ("product_card", "email_pair"):
                result[f"{name}_prior_support"] = support
        return result
