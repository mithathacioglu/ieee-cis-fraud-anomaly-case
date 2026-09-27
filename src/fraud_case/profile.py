"""Case'in ilk iki adımını indirilen eğitim tabloları üzerinde çalıştırır."""

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fraud_case.data import chronological_split, load_data
from fraud_case.profiling import (
    categorical_distributions,
    column_profiles,
    entity_patterns,
    numeric_relationships,
    rare_combinations,
)


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def charts(train: pd.DataFrame, columns: pd.DataFrame, entities: pd.DataFrame, destination: Path) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    axes[0, 0].hist(np.log1p(train["TransactionAmt"]), bins=60, color="#256580")
    axes[0, 0].set(title="Transaction amount", xlabel="log(1 + amount), original units", ylabel="Transactions")
    missing = columns.set_index("column")["null_ratio"].nlargest(12).sort_values()
    axes[0, 1].barh(missing.index, missing.values * 100, color="#ae694c")
    axes[0, 1].set(title="Fields with most missing values", xlabel="Missing (%)", xlim=(0, 100))
    days = (train["TransactionDT"] // 86400).astype(int)
    daily = train.groupby(days).agg(transactions=("TransactionID", "size"))
    axes[1, 0].plot(daily.index, daily["transactions"], color="#256580")
    axes[1, 0].set(title="Transaction volume over relative time", xlabel="Relative day (not a calendar date)", ylabel="Transactions")
    axes[1, 0].scatter(daily.index[-1], daily["transactions"].iloc[-1], color="#ae694c", zorder=3, label="Last day ends at train cutoff")
    axes[1, 0].legend(frameon=False, fontsize=8)
    if len(entities):
        axes[1, 1].hist(np.log10(entities["transaction_count"]), bins=40, color="#537b51")
    axes[1, 1].set(title="Transactions per complete entity proxy", xlabel="log10(transaction count)", ylabel="Entity proxies")
    fig.suptitle(f"IEEE-CIS training partition only | {len(train):,} transactions", fontsize=15)
    fig.savefig(destination / "data_profile.png", dpi=150)
    plt.close(fig)


def report(summary: dict, columns: pd.DataFrame) -> str:
    quality = summary["quality"]
    train = summary["splits"]["train"]
    high = columns.loc[columns["high_cardinality"], ["column", "nunique"]].sort_values("nunique", ascending=False)
    missing = columns.sort_values("null_ratio", ascending=False).head(10)
    lines = [
        "# Veri hazırlama ve profil analizi", "",
        "Kaynak: [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data).", "",
        "## Kapsam", "",
        "Birleştirme ve temel kalite kontrolleri tüm eğitim dosyalarını kapsar. Dağılımlar, korelasyonlar, nadir "
        "kombinasyonlar ve entity örüntüleri yalnızca zaman sırasındaki ilk %60 üzerinde hesaplanır. Sonraki %20 "
        "validation, son %20 test için ayrılır. Eşit zaman damgaları farklı bölümlere dağıtılmaz.", "",
        "Bu rapor betikle üretilir. Veri kalite kontrolleri dağılımı değiştirmez; doldurma, outlier silme veya model eğitimi yapılmaz.", "",
        "## Birleştirme ve kalite", "",
        f"- Transaction: {quality['transaction_rows']:,} satır; identity: {quality['identity_rows']:,} satır.",
        f"- Left join sonucu: {quality['joined_rows']:,} satır, {quality['joined_columns']} kolon (`has_identity` dahil).",
        f"- Identity eşleşmesi: %{quality['identity_coverage'] * 100:.2f}; eşleşmeyen işlem: {quality['rows_without_identity']:,}.",
        f"- Tüm dosyadaki fraud oranı: %{quality['fraud_rate'] * 100:.2f} ({quality['fraud_rows']:,} işlem).",
        "- Tekrarlı/eksik anahtar, orphan identity, negatif veya sonlu olmayan zaman/tutar ve geçersiz etiket kontrolleri geçti.",
        f"- Sıfır tutarlı işlem: {quality['zero_amount_rows']:,}. Bu değer eksik tutarla aynı kabul edilmez.", "",
        "## Zaman ayrımı", "", "| Bölüm | Satır | İlk saniye | Son saniye |", "|---|---:|---:|---:|",
    ]
    for name, part in summary["splits"].items():
        lines.append(f"| {name} | {part['rows']:,} | {part['min_time']} | {part['max_time']} |")
    lines.extend([
        "", "`TransactionDT` açıklanmayan bir referansa göre geçen saniyedir. Saat, haftanın günü ve hafta sonu için gerçek takvim varsayımı burada yapılmaz. `elapsed_time` rolü, gerçek `datetime` rolünden ayrıdır.", "",
        "## Kolon tipleri ve eksiklik", "",
        "Fiziksel dtype ile semantik rol ayrılır. Örneğin `card1` sayısal saklanan bir kategoridir; büyüklüğü kartlar "
        "arasında anlamlı mesafe oluşturmaz. ID ve hedef ayrı roller alır. Tanınmayan numeric kolonlara isimlerinden iş "
        "anlamı yüklenmez.", "",
        f"Train profilinde {train['rows']:,} satır ve {len(columns)} kolon var. Rol dağılımı: " + ", ".join(f"{k}: {v}" for k, v in columns["role"].value_counts().items()) + ".", "",
        "| Kolon | Eksik oranı |", "|---|---:|",
    ])
    lines.extend(f"| {row.column} | %{row.null_ratio * 100:.2f} |" for row in missing.itertuples())
    lines.extend(["", "Eksik değerler bu aşamada korunur. Identity tablosunun bulunmaması ile bulunan identity kaydındaki boş bir alan, `has_identity` sayesinde ayrıştırılabilir.", "", "## Yüksek cardinality", "",
                  "Kategorik alanlarda en az 100 farklı değer veya en az 20 farklı değerle birlikte %5 benzersizlik oranı "
                  "kullanılır. Bu bir inceleme eşiğidir; kolon silme kararı değildir.", "", "| Kolon | Farklı değer |", "|---|---:|"])
    lines.extend(f"| {row.column} | {row.nunique:,} |" for row in high.itertuples())
    lines.extend(["", "## Dağılım ve outlier", "",
        "Her numeric alan için min, p01, p25, medyan, p75, p99, max, ortalama, standart sapma ve IQR kaydedilir. IQR "
        "sınırı dışındaki bir değer otomatik olarak fraud sayılmaz. IQR=0 olan ayrık dağılımlar ayrıca işaretlenir.", "",
        "![Train veri profili](figures/data_profile.png)", "",
        "Zaman grafiğindeki son gün train sınırında kesilir. O noktadaki düşük hacim tam günlük bir düşüş olarak yorumlanmamalıdır.", "",
        "## Nadir kombinasyonlar", "",
        "Birlikte gözlenen kategorilerde destek sayısı en fazla 5 olan kombinasyonlar raporlanır. Eksik bileşenler bu "
        "hesaptan çıkarılır; sayıları ayrıca verilir. Bu betimleyici özetler gelecekteki işlemler için doğrudan feature "
        "değildir.", "",
    ])
    for name, combination in summary["rare_combinations"].items():
        lines.append(f"- `{name}`: {combination['distinct_combinations']:,} farklı kombinasyon; {combination['rare_combinations']:,} nadir kombinasyon, {combination['rare_rows']:,} satır. Eksik bileşen nedeniyle dışarıda kalan: {combination['excluded_missing_rows']:,}.")
    rel = summary["relationships"]
    lines.extend(["", "## Kolon ilişkileri", "", f"Train içinden sabit tohumla seçilen {rel['sample_rows']:,} satırda Spearman korelasyonu kullanılır. Her çift için en az {rel['min_pair_support']} ortak dolu gözlem gerekir. Hedef, kimlik ve kategori kodları dahil edilmez. Korelasyon neden-sonuç ilişkisi kurmaz.", "", "| Alan 1 | Alan 2 | Spearman | Ortak gözlem |", "|---|---|---:|---:|"])
    for pair in rel["top_pairs"][:10]:
        lines.append(f"| {pair['left']} | {pair['right']} | {pair['spearman']:.3f} | {pair['paired_rows']:,} |")
    entity = summary["entities"]
    lines.extend(["", "## Entity davranışları", "",
        "Vekil anahtar: `card1 + card2 + card3 + card5 + addr1`. Gerçek kullanıcı kimliği değildir; çakışma ve aynı "
        "kişinin farklı anahtarlara dağılması mümkündür. Bileşenleri eksik işlemler tek bir 'unknown user' grubunda "
        "birleştirilmez.", "",
        f"- Tam anahtarlı işlem: {entity['eligible_rows']:,}; eksik anahtarlı işlem: {entity['excluded_incomplete_rows']:,}.",
        f"- Entity vekili sayısı: {entity['entities']:,}; yalnızca bir işlemli: {entity['single_transaction_entities']:,}.",
        f"- Entity başına medyan işlem sayısı: {entity['transaction_count_quantiles']['0.5']:.0f}.",
        f"- Ardışık işlemler arasındaki medyan süre: {entity['median_gap_seconds']} saniye.", "",
        "Entity başına işlem sayısı, ortalama/std tutar, faaliyet süresi, ürün ve e-posta alanı çeşitliliği hesaplanır. "
        "Bunlar train döneminin betimleyici özetleridir. Model feature'larında her işlem için yalnızca önceki kayıtlar "
        "kullanılacaktır.", "",
        "## Üretilen dosyalar", "",
        "- `artifacts/profile/summary.json`: kalite, split, nadir kombinasyon, ilişki ve entity özetleri.",
        "- `artifacts/profile/columns.json`: tüm kolonların tip, eksik değer, cardinality ve dağılım profili.",
        "- `artifacts/profile/categories.json`: kategori frekansları ve eksik sayıları.",
        "- `data/processed/transactions.parquet`: satır kaybetmeden birleştirilmiş veri ve split etiketi.",
        "- `data/processed/entity_profile.parquet`: yalnızca train entity özetleri; model girdisi olarak kullanılamaz.", "",
        "Dosya parmak izleri `data/raw/manifest.json` içindedir. Ham veri ve satır bazlı türevler Git kapsamı dışındadır.", "",
    ])
    return "\n".join(lines)


def run(root: Path) -> None:
    output = root / "artifacts" / "profile"
    processed = root / "data" / "processed"
    reports = root / "reports"
    figures = reports / "figures"
    for directory in (output, processed, figures):
        directory.mkdir(parents=True, exist_ok=True)
    print("Loading and validating both tables ...", flush=True)
    data, quality = load_data(root / "data" / "raw")
    split = chronological_split(data)
    split_summary = {}
    for name in ("train", "validation", "test"):
        part = data.loc[split.eq(name), "TransactionDT"]
        split_summary[name] = {"rows": len(part), "min_time": int(part.min()), "max_time": int(part.max())}
    print("Writing joined data with chronological split ...", flush=True)
    data.assign(split=split).to_parquet(processed / "transactions.parquet", index=False)
    train = data.loc[split.eq("train")]
    print(f"Profiling {len(train):,} train rows ...", flush=True)
    columns = column_profiles(train)
    categories = categorical_distributions(train, columns)
    rare = rare_combinations(train)
    print("Measuring column relationships and entity patterns ...", flush=True)
    relationships = numeric_relationships(train, columns)
    entities, entity_table = entity_patterns(train)
    entity_table.to_parquet(processed / "entity_profile.parquet")
    summary = {"quality": quality, "splits": split_summary, "rare_combinations": rare,
               "relationships": relationships, "entities": entities,
               "runtime": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__}}
    columns.to_json(output / "columns.json", orient="records", indent=2)
    write_json(output / "categories.json", categories)
    write_json(output / "summary.json", summary)
    charts(train, columns, entity_table, figures)
    (reports / "data_profile.md").write_text(report(summary, columns), encoding="utf-8")
    print(json.dumps({"quality": quality, "splits": split_summary}, indent=2), flush=True)
    print("Report: reports/data_profile.md", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    run(args.root.resolve())


if __name__ == "__main__":
    main()
