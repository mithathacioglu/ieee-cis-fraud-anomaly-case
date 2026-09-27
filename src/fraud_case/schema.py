"""card1 sayı olarak saklanıyor ama kategori kodu. dtype rol demek değil."""

import re

import pandas as pd

CODE_COLUMNS = {f"card{i}" for i in range(1, 7)} | {"addr1", "addr2"} | {f"id_{i:02d}" for i in range(12, 39)}


def column_role(name: str, values: pd.Series) -> tuple[str, str]:
    if name == "TransactionID":
        return "identifier", "join key; excluded from model features"
    if name == "isFraud":
        return "target", "evaluation label; excluded from anomaly inputs"
    if name == "TransactionDT":
        return "elapsed_time", "seconds from an undisclosed reference; not a calendar timestamp"
    if name in CODE_COLUMNS or re.fullmatch(r"M[1-9]", name):
        return "categorical", "dataset code field; numeric storage does not imply magnitude"
    if pd.api.types.is_datetime64_any_dtype(values):
        return "datetime", "native datetime dtype"
    if pd.api.types.is_bool_dtype(values):
        return "categorical", "boolean"
    if pd.api.types.is_numeric_dtype(values):
        return "numeric", "numeric dtype; low-cardinality values remain candidates for review"
    sample = values.dropna().astype(str).head(200)
    if len(sample) and sample.str.match(r"^\d{4}-\d{2}-\d{2}(?:[ T].*)?$").all():
        parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
        if parsed.notna().all():
            return "datetime", "ISO-shaped strings parsed in a sample; confirm before conversion"
    return "categorical", "string or category dtype"
