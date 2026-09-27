"""features.parquet üretir.

Diskten sadece REQUIRED_COLUMNS okunuyor, yani isFraud buraya hiç girmiyor.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_case.features import FEATURE_SPECS, REQUIRED_COLUMNS, FeatureBuilder


def run(root: Path, calendar_reference: str | None = None) -> None:
    start = time.perf_counter()
    directory = root / "artifacts" / "features"
    directory.mkdir(parents=True, exist_ok=True)
    data = pd.read_parquet(root / "data" / "processed" / "transactions.parquet", columns=REQUIRED_COLUMNS + ["split"])
    builder = FeatureBuilder(calendar_reference=calendar_reference)
    outputs = []
    summary = {"calendar_reference": calendar_reference, "history_mode": "chronological_unlabelled_updates", "feature_count": len(FEATURE_SPECS), "splits": {}}
    for split in ("train", "validation", "test"):
        batch = data.loc[data["split"].eq(split)]
        print(f"Building {split}: {len(batch):,} transactions ...", flush=True)
        features = builder.transform(batch)
        if not features.TransactionID.equals(batch.TransactionID.reset_index(drop=True)):
            raise AssertionError("Feature IDs are misaligned")
        numeric = features[list(FEATURE_SPECS)]
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise AssertionError("An infinite feature was produced")
        summary["splits"][split] = {
            "rows": len(features),
            "entity_known_ratio": float(features.entity_known.mean()),
            "history_sufficient_ratio": float(features.history_sufficient.mean()),
            "feature_null_ratio": {name: float(value) for name, value in numeric.isna().mean().items()},
        }
        features["split"] = split
        outputs.append(features)
    combined = pd.concat(outputs, ignore_index=True)
    combined.to_parquet(root / "data" / "processed" / "features.parquet", index=False)
    summary["elapsed_seconds"] = round(time.perf_counter() - start, 3)
    (directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    contract = {name: {"group": group, "definition": definition} for name, (group, definition) in FEATURE_SPECS.items()}
    (directory / "contract.json").write_text(json.dumps(contract, indent=2) + "\n", encoding="utf-8")
    lines = ["# Feature engineering", "", f"{len(combined):,} işlem için {len(FEATURE_SPECS)} özellik üretildi.", "",
             "Hedef (`isFraud`) dosyadan dahi okunmaz. `TransactionID` ve `split` yalnızca eşleme metadatasıdır; feature sözleşmesinde yoktur.", "",
             "Her zaman grubunda önce bütün feature'lar hesaplanır, ardından geçmiş güncellenir. Validation ve test "
             "sırasında da yalnızca daha önce gözlenen etiketsiz işlemler geçmişi günceller. Model parametrelerinin bu "
             "bölümlerde yeniden fit edilmesi anlamına gelmez.", "",
             "## Takvim bağlamı", "", f"Açık başlangıç referansı: `{calendar_reference}`.", "",
             "Referans yoksa gerçek hour/day_of_week/is_weekend/is_business_hours alanları null kalır. Relative day, 24 "
             "saatlik faz ve sin/cos alanları yine kullanılabilir. Açık referans verilirse takvim hesapları bu varsayıma "
             "dayanır; IEEE-CIS tarafından doğrulanmış tarih olarak sunulamaz.", "",
             "## Kapsama", "", "| Bölüm | İşlem | Tam entity vekili | En az 5 geçmiş işlem |", "|---|---:|---:|---:|"]
    for split, part in summary["splits"].items():
        lines.append(f"| {split} | {part['rows']:,} | %{part['entity_known_ratio'] * 100:.2f} | %{part['history_sufficient_ratio'] * 100:.2f} |")
    lines += ["", "## Özellik sözleşmesi", "", "| Özellik | Grup | Tanım |", "|---|---|---|"]
    for name, (group, definition) in FEATURE_SPECS.items():
        lines.append(f"| `{name}` | {group} | {definition} |")
    lines += ["", "## Sınırlar", "",
              "Entity, önceki adımlardaki kart/adres vekilidir. DeviceInfo benzersiz bir cihaz ID'si değil, paylaşılan cihaz "
              "açıklaması olabilir. E-posta alanı değişimi kişinin e-posta adresinin değiştiğini kanıtlamaz.", "",
              "Geçmişsiz entity için count=0 ve ortalama=null; entity tanımsızsa count da null'dır. İlk gözlem yeni ilişki "
              "anomalisine dönüştürülmez. Sık işlem yapmak güvenilirlik etiketi değildir. Skorlama katmanında minimum geçmiş "
              "ve eksiklikler ayrıca ele alınmalıdır.", "",
              "Velocity pencereleri sol sınırı dahil, şimdiki zamanı hariç tutar: [t-3600,t) ve [t-86400,t).", "",
              "FeatureBuilder tek yazarlı bir stream nesnesidir. Yeni batch önceki batch'in son zamanından büyük bir zamanda "
              "başlamalıdır; aynı saniye iki batch'e bölünemez. Geç gelen olaylar reddedilir; yeniden sıralama/geri sarma bu "
              "prototipte yoktur.", ""]
    (root / "reports" / "features.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(combined):,} rows, {len(FEATURE_SPECS)} features, {summary['elapsed_seconds']} seconds.", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--calendar-reference", help="Explicit scenario reference including UTC offset; omitted by default")
    args = parser.parse_args()
    run(args.root.resolve(), args.calendar_reference)


if __name__ == "__main__":
    main()
