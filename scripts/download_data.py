"""İki IEEE-CIS eğitim dosyasını indirir ve içeriğini doğrular."""

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
COMPETITION = "ieee-fraud-detection"
FILES = ("train_transaction.csv", "train_identity.csv")


def download(directory: Path) -> None:
    cli = Path(sys.executable).with_name("kaggle.exe" if sys.platform == "win32" else "kaggle")
    if not cli.is_file():
        raise RuntimeError("Install kaggle in this Python environment first: python -m pip install kaggle")
    for name in FILES:
        target = directory / name
        if target.is_file():
            print(f"Found {name}; verifying existing file below.", flush=True)
            continue
        archive = directory / f"{name}.zip"
        if not archive.is_file():
            subprocess.run(
                [str(cli), "competitions", "download", COMPETITION, "-f", name, "-p", str(directory)],
                check=True,
            )
        if archive.is_file():
            # Arşivden sadece beklenen dosyayı çıkar, rastgele yol açma
            with zipfile.ZipFile(archive) as source:
                member = source.getinfo(name)
                temporary = directory / f"{name}.partial"
                with source.open(member) as incoming, temporary.open("wb") as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                temporary.replace(target)
        if not target.is_file():
            raise RuntimeError(f"Download did not produce {name}")


def inspect_file(path: Path) -> tuple[dict, set[int]]:
    print(f"Checking {path.name} ...", flush=True)
    with path.open("rb") as stream:
        checksum = hashlib.file_digest(stream, "sha256").hexdigest()
    ids: set[int] = set()
    labels: Counter = Counter()
    row_count = 0
    with path.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream)
        header = next(reader)
        if len(header) != len(set(header)):
            raise ValueError(f"Duplicate column names in {path.name}")
        id_index = header.index("TransactionID")
        label_index = header.index("isFraud") if path.name == FILES[0] else None
        if label_index is not None:
            for required in ("TransactionDT", "TransactionAmt"):
                if required not in header:
                    raise ValueError(f"Missing {required} in {path.name}")
        for row_count, row in enumerate(reader, start=1):
            if len(row) != len(header):
                raise ValueError(f"Wrong column count in {path.name}, data row {row_count}")
            transaction_id = int(row[id_index])
            if transaction_id in ids:
                raise ValueError(f"Duplicate TransactionID in {path.name}: {transaction_id}")
            ids.add(transaction_id)
            if label_index is not None:
                label = row[label_index]
                if label not in ("0", "1"):
                    raise ValueError(f"Invalid isFraud value at data row {row_count}")
                labels[label] += 1
    if not row_count:
        raise ValueError(f"Empty data file: {path.name}")
    details = {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": checksum,
        "rows": row_count,
        "columns": len(header),
        "unique_transaction_ids": len(ids),
    }
    if labels:
        details["isFraud_counts"] = dict(sorted(labels.items()))
    return details, ids


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true", help="Check files downloaded through the browser")
    args = parser.parse_args()
    directory = ROOT / "data" / "raw"
    directory.mkdir(parents=True, exist_ok=True)
    if not args.verify_only:
        download(directory)
    transactions, transaction_ids = inspect_file(directory / FILES[0])
    identity, identity_ids = inspect_file(directory / FILES[1])
    orphan_count = len(identity_ids - transaction_ids)
    if orphan_count:
        raise ValueError(f"Identity contains {orphan_count} IDs absent from transactions")
    manifest = {
        "source": f"https://www.kaggle.com/competitions/{COMPETITION}/data",
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
        "files": [transactions, identity],
        "transactions_with_identity": len(identity_ids),
        "transactions_without_identity": len(transaction_ids - identity_ids),
    }
    (directory / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
