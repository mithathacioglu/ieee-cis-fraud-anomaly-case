import numpy as np
import pandas as pd

from fraud_case.data import entity_proxy
from fraud_case.schema import column_role


def column_profiles(frame: pd.DataFrame) -> pd.DataFrame:
    records = []
    for name, values in frame.items():
        role, reason = column_role(name, values)
        non_null = int(values.notna().sum())
        unique = int(values.nunique())
        record = {
            "column": name, "dtype": str(values.dtype), "role": role, "reason": reason,
            "null_ratio": float(values.isna().mean()), "nunique": unique,
            "unique_ratio_non_null": unique / non_null if non_null else None,
            "high_cardinality": role == "categorical" and (unique >= 100 or (unique >= 20 and unique / max(non_null, 1) >= 0.05)),
            "constant": unique <= 1, "infinite_count": 0,
        }
        if role in {"numeric", "elapsed_time"}:
            finite = values[np.isfinite(values)]
            record["infinite_count"] = int(np.isinf(values).sum())
            if len(finite):
                q01, q25, q50, q75, q99 = finite.quantile([0.01, 0.25, 0.5, 0.75, 0.99])
                iqr = q75 - q25
                # IQR=0 ayrıca işaretleniyor; fraud kanıtı değil
                outliers = (finite < q25 - 1.5 * iqr) | (finite > q75 + 1.5 * iqr)
                record.update({
                    "min": float(finite.min()), "p01": float(q01), "p25": float(q25),
                    "median": float(q50), "p75": float(q75), "p99": float(q99),
                    "max": float(finite.max()), "mean": float(finite.mean()),
                    "std": float(finite.std(ddof=0)), "iqr": float(iqr),
                    "iqr_outlier_ratio": float(outliers.mean()), "zero_iqr": bool(iqr == 0),
                })
        records.append(record)
    return pd.DataFrame(records)


def categorical_distributions(frame: pd.DataFrame, schema: pd.DataFrame) -> dict:
    result = {}
    for name in schema.loc[schema["role"].eq("categorical"), "column"]:
        counts = frame[name].value_counts(dropna=True)
        result[name] = {
            "missing_count": int(frame[name].isna().sum()),
            "top_values": [{"value": str(k), "count": int(v)} for k, v in counts.head(10).items()],
            "other_non_null_count": int(counts.iloc[10:].sum()),
        }
    return result


def rare_combinations(frame: pd.DataFrame, max_support: int = 5) -> dict:
    result = {}
    for columns in (["ProductCD", "card4"], ["P_emaildomain", "R_emaildomain"], ["DeviceType", "id_31"]):
        # Eksiklik başka yerde ölçülüyor; nadir örüntü için değer gözlenmiş olmalı
        observed = frame[list(columns)].dropna()
        counts = observed.value_counts()
        rare = counts[counts <= max_support].sort_values(kind="stable")
        result[" + ".join(columns)] = {
            "max_support": max_support, "eligible_rows": len(observed),
            "excluded_missing_rows": len(frame) - len(observed),
            "distinct_combinations": len(counts), "rare_combinations": len(rare),
            "rare_rows": int(rare.sum()),
            "examples": [{"values": [str(x) for x in key], "count": int(count)} for key, count in rare.head(15).items()],
        }
    return result


def numeric_relationships(frame: pd.DataFrame, schema: pd.DataFrame, sample_size: int = 30000) -> dict:
    columns = schema.loc[schema["role"].eq("numeric") & ~schema["constant"] & (schema["null_ratio"] < 0.95), "column"].tolist()
    sample = frame[columns].sample(n=min(sample_size, len(frame)), random_state=42).replace([np.inf, -np.inf], np.nan)
    correlations = sample.corr(method="spearman", min_periods=100)
    pairs = []
    # Çift bazlı dolu gözlem sayısını da ver, yoksa korelasyon yanıltır
    present = sample.notna().to_numpy(dtype=np.int32)
    supports = present.T @ present
    for i, left in enumerate(columns):
        for j in range(i + 1, len(columns)):
            value = correlations.iloc[i, j]
            if np.isfinite(value):
                pairs.append({"left": left, "right": columns[j], "spearman": float(value), "paired_rows": int(supports[i, j])})
    pairs.sort(key=lambda p: (-abs(p["spearman"]), p["left"], p["right"]))
    return {"method": "spearman", "sample_rows": len(sample), "seed": 42, "min_pair_support": 100, "top_pairs": pairs[:30]}


def entity_patterns(frame: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    proxy = entity_proxy(frame)
    fields = ["TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "P_emaildomain"]
    eligible = frame[fields].assign(entity_proxy=proxy).dropna(subset=["entity_proxy"])
    grouped = eligible.groupby("entity_proxy", observed=True)
    entities = grouped.agg(
        transaction_count=("TransactionID", "size"), amount_mean=("TransactionAmt", "mean"),
        amount_std=("TransactionAmt", "std"), first_seen=("TransactionDT", "min"), last_seen=("TransactionDT", "max"),
        distinct_products=("ProductCD", "nunique"), distinct_email_domains=("P_emaildomain", "nunique"),
    )
    entities["active_days"] = (entities["last_seen"] - entities["first_seen"]) / 86400
    ordered = eligible.sort_values(["TransactionDT", "TransactionID"])
    gaps = ordered.groupby("entity_proxy", observed=True)["TransactionDT"].diff().dropna()
    quantiles = entities["transaction_count"].quantile([0.5, 0.9, 0.99])
    summary = {
        "proxy_columns": ["card1", "card2", "card3", "card5", "addr1"],
        "limitation": "Shared card/address codes can combine different people; this is not a user ID.",
        "eligible_rows": int(proxy.notna().sum()), "excluded_incomplete_rows": int(proxy.isna().sum()),
        "entities": len(entities), "single_transaction_entities": int(entities["transaction_count"].eq(1).sum()),
        "transaction_count_quantiles": {str(k): float(v) for k, v in quantiles.items()},
        "median_gap_seconds": float(gaps.median()) if len(gaps) else None,
        "same_timestamp_gaps": int(gaps.eq(0).sum()),
    }
    return summary, entities
