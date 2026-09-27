"""Etiket kullanan iki modelle karşılaştırma.

Dört anomali katmanı etiket okumuyor. Bu betik aynı kronolojik split'i
kullanan iki gözetimli modeli train'de eğitip sonraki
validation yarısında karşılaştırır.

Üç kontrol var:
  - eşlenmiş satır bootstrap aralığı. Yeniden
    örnekleme satır bazlı; aynı entity'nin işlemleri ve zaman bağımlılığı
    korunmuyor; kapsam hatasının yönü ve büyüklüğü bu deneyden bilinmiyor
  - monoton zaman taşıyan feature'lar (prior_global_count, relative_day)
    çıkarılmış duyarlılık koşusu
  - iki gözetimli modelde de balanced sınıf ağırlığı

Karşılaştırma girdi bakımından eşlenmiş değil: gözetimli modeller feature
sözleşmesinin tamamını görüyor, gözetimsiz katmanlar daha dar bir alt küme
kullanıyor. Yani fark yalnız etiket kullanmaktan gelmiyor.

Final test setine dokunulmuyor; o bölüm bir kez açıldı ve kapandı.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fraud_case.anomaly import MULTIVARIATE_FEATURES, REQUIRED_FEATURES
from fraud_case.evaluate_context import validation_sections
from fraud_case.evaluation import top_k_hits
from fraud_case.features import FEATURE_SPECS

ROOT = Path(__file__).resolve().parents[1]
BUDGETS = [.01, .05, .10]
BOOTSTRAP = 400
SEED = 42
# Monoton artan, yani train/validation kaymasini tasiyabilecek feature'lar.
# Tek basina prior_global_count'u cikarmak yetmez: relative_day de ayni bilgiyi tasiyor.
TIME_PROXIES = ["prior_global_count", "relative_day"]
NAMES = {"unsupervised_raw": "Gözetimsiz raw skor", "logistic_regression": "Logistic regression",
         "gradient_boosting": "Gradient boosting"}


# top_k_hits ve blok bootstrap fraud_case.evaluation icinde: hem burada hem
# kronolojik deneyde hem urun context belirsizliginde ayni fonksiyon kullaniliyor.


def evaluate(y, scores, ids):
    row = {"average_precision": float(average_precision_score(y, scores)),
           "roc_auc": float(roc_auc_score(y, scores)), "budgets": {}}
    for rate in BUDGETS:
        count = int(np.ceil(len(y) * rate))
        row["budgets"][str(rate)] = {"reviews": count, "tp": top_k_hits(y, scores, ids, count)}
    return row


def bootstrap_gap(y, baseline, challenger, ids, rate=.05):
    """Paired IID row bootstrap, conditional on this window and fitted models.

    Both rankings use the same resampled rows. Entity/time dependence and
    model fitting uncertainty are not represented; this is not evidence of
    performance on a future period.
    """
    rng = np.random.default_rng(SEED)
    n = len(y)
    count = int(np.ceil(n * rate))
    gaps = np.empty(BOOTSTRAP, dtype=int)
    for i in range(BOOTSTRAP):
        pick = rng.integers(0, n, n)
        gaps[i] = (top_k_hits(y[pick], challenger[pick], ids[pick], count)
                   - top_k_hits(y[pick], baseline[pick], ids[pick], count))
    low, high = np.percentile(gaps, [2.5, 97.5])
    return {"observed_gap": top_k_hits(y, challenger, ids, count) - top_k_hits(y, baseline, ids, count),
            "ci_low": float(low), "ci_high": float(high), "resamples": BOOTSTRAP,
            "excludes_zero": bool(low > 0 or high < 0)}


def fit_rank(columns, x_train, y_train, x_audit, *, early_stopping=False):
    """Train two reference models with balanced class weights.

    early_stopping varsayilan olarak kapali: "auto" bu boyuttaki veride sklearn'i
    True'ya cevirir ve egitim setinden RASTGELE %10 ic validation ayirir. Kronolojik
    split'e bu kadar dikkat edilen bir yerde rastgele bolme tutarsiz olur.
    """
    models = {
        "logistic_regression": make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        ),
        "gradient_boosting": HistGradientBoostingClassifier(
            class_weight="balanced", random_state=SEED, early_stopping=early_stopping),
    }
    ranked = {}
    for name, model in models.items():
        print(f"  {name}: {len(x_train):,} satir, {len(columns)} feature ...", flush=True)
        model.fit(x_train[columns], y_train)
        ranked[name] = model.predict_proba(x_audit[columns])[:, 1]
    return ranked


def main(data_root=None, output=None):
    data_root = data_root or ROOT / "data/processed"
    keep = [("split", "in", ["train", "validation"])]
    features = pd.read_parquet(data_root / "features.parquet", filters=keep).reset_index(drop=True)
    scores = pd.read_parquet(data_root / "scores.parquet", filters=keep).reset_index(drop=True)
    source = pd.read_parquet(data_root / "transactions.parquet",
                             columns=["TransactionID", "TransactionDT", "isFraud", "split"],
                             filters=keep).reset_index(drop=True)
    assert features.TransactionID.equals(source.TransactionID)
    assert features.TransactionID.equals(scores.TransactionID)

    train = source["split"].eq("train").to_numpy()
    # Takvim referansi verilmedigi icin hour/day_of_week/is_weekend/is_business_hours
    # train'de tamamen null, calendar_available sabit. Bilgi tasimayan kolonlar elenir.
    pool = features.loc[train, list(FEATURE_SPECS)]
    dropped = sorted(c for c in FEATURE_SPECS if pool[c].isna().all() or pool[c].nunique(dropna=True) <= 1)
    columns = [c for c in FEATURE_SPECS if c not in dropped]
    print("elenen kolonlar: " + ", ".join(dropped), flush=True)

    # Sonraki validation yarisi: butun diger raporlar da bu bolumu kullaniyor.
    # TransactionDT features.parquet'te yok, kaynak tablodan aliniyor.
    validation = features.loc[~train, ["TransactionID"]].copy()
    validation["TransactionDT"] = source.loc[~train, "TransactionDT"].to_numpy()
    _, audit, _ = validation_sections(validation.reset_index(drop=True), .5)
    audit_index = features.index[~train][audit.to_numpy()]

    y = source.loc[audit_index, "isFraud"].to_numpy()
    ids = source.loc[audit_index, "TransactionID"].to_numpy()
    x_train, y_train = features.loc[train], source.loc[train, "isFraud"]
    x_audit = features.loc[audit_index]

    print("tam feature kumesi:", flush=True)
    ranked = {"unsupervised_raw": scores.loc[audit_index, "raw_anomaly_score"].to_numpy()}
    ranked.update(fit_rank(columns, x_train, y_train, x_audit))
    result = {name: evaluate(y, values, ids) for name, values in ranked.items()}

    print(f"{', '.join(TIME_PROXIES)} cikarilmis duyarlilik kosusu:", flush=True)
    without = [c for c in columns if c not in TIME_PROXIES]
    sensitivity = {name: evaluate(y, values, ids)
                   for name, values in fit_rank(without, x_train, y_train, x_audit).items()}

    gaps = {name: bootstrap_gap(y, ranked["unsupervised_raw"], values, ids)
            for name, values in ranked.items() if name != "unsupervised_raw"}

    summary = {
        "scope": "Later validation half. Train labels only; the held-out test split was not reopened.",
        "rows": len(audit_index), "positives": int(y.sum()),
        "features_used": len(columns), "features_dropped": dropped,
        "feature_columns": columns,
        "unsupervised_inputs": len(REQUIRED_FEATURES), "isolation_forest_inputs": len(MULTIVARIATE_FEATURES),
        "input_parity": "Not matched: the supervised models see every usable feature, the anomaly layers a subset.",
        "class_weight": "balanced for both supervised models",
        "models": result, "bootstrap_gap_at_5pct": gaps,
        "bootstrap_scope": "Paired IID row resampling, conditional on fitted models and this window; "
                           "time/entity dependence and refitting uncertainty are not represented.",
        "sensitivity_without_time_proxy": {"removed": TIME_PROXIES, "models": sensitivity},
        "interpretation": "Inputs are not matched. The gap cannot be attributed solely to labels. "
                          "Supervised training assumes labels are available; actual label availability is unknown. "
                          "Neither future performance nor incremental hybrid benefit is established here.",
    }
    out = output or ROOT / "artifacts/baseline"
    out.mkdir(parents=True, exist_ok=True)
    (out / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    render(summary)
    print(json.dumps({k: {"ap": round(v["average_precision"], 4), "auc": round(v["roc_auc"], 4),
                          "tp_5pct": v["budgets"]["0.05"]["tp"]} for k, v in result.items()}, indent=2), flush=True)


def sensitivity_statement(name, full, ablated):
    """Describe all reported point estimates without a significance claim."""
    deltas = [ablated[key] - full[key] for key in ("average_precision", "roc_auc")]
    deltas += [ablated["budgets"][str(rate)]["tp"] - full["budgets"][str(rate)]["tp"] for rate in BUDGETS]
    up, down = sum(d > 0 for d in deltas), sum(d < 0 for d in deltas)
    if up and down:
        direction = "metrikler farklı yönlerde değişti"
    elif up:
        direction = "bazı metrikler yükseldi, hiçbir metrik düşmedi"
    elif down:
        direction = "bazı metrikler düştü, hiçbir metrik yükselmedi"
    else:
        direction = "raporlanan metrikler değişmedi"
    return f"{name}: {direction} ({up} artış, {down} düşüş, {len(deltas) - up - down} eşitlik)."


def render(summary, *, output_path=None, evidence_source="artifacts/baseline/evaluation.json"):
    result, gaps = summary["models"], summary["bootstrap_gap_at_5pct"]
    raw, gb = result["unsupervised_raw"], result["gradient_boosting"]
    budget = raw["budgets"]["0.05"]
    lines = ["# Gözetimli baseline karşılaştırması", "",
             "İki gözetimli model, dört anomali katmanının birleşik skoruyla aynı değerlendirme döneminde karşılaştırıldı.", "",
             f"Aynı kronolojik split. Takvim referansı verilmediği için train'de tamamen boş veya sabit kalan "
             f"{len(summary['features_dropped'])} kolon elendi "
             f"({', '.join('`'+d+'`' for d in summary['features_dropped'])}); gözetimli modeller kalan "
             f"{summary['features_used']} feature'ı görüyor. Karşılaştırma sonraki validation yarısının "
             f"{summary['rows']:,} satırında, {summary['positives']:,} gerçek fraud üzerinde yapıldı. İki gözetimli "
             "model de aynı sınıf ağırlığıyla eğitildi. Final test bölümü açılmadı.", "",
             f"**Girdiler eşlenmiş değil.** Gözetimli modeller {summary['features_used']} feature görüyor; dört "
             f"anomali katmanının toplam girdisi {summary['unsupervised_inputs']}, IsolationForest'ınki "
             f"{summary['isolation_forest_inputs']}. Yani aşağıdaki fark yalnız etiket kullanmaktan gelmiyor, "
             "girdi kümesi farkı da içinde. Kontrollü bir deney değil, bir referans noktası.", "",
             "| Yöntem | Etiket ister | AP | ROC AUC |", "|---|---|---:|---:|"]
    for key, row in result.items():
        needs = "hayır" if key == "unsupervised_raw" else "evet"
        lines.append(f"| {NAMES[key]} | {needs} | {row['average_precision']:.4f} | {row['roc_auc']:.4f} |")
    lines += ["", "## Aynı inceleme bütçesinde yakalanan fraud", "",
              "| Bütçe | İnceleme | " + " | ".join(NAMES[k] for k in result) + " |",
              "|---|---:|" + "---:|" * len(result)]
    for rate in BUDGETS:
        cells = " | ".join(str(result[k]["budgets"][str(rate)]["tp"]) for k in result)
        lines.append(f"| %{rate*100:.0f} | {raw['budgets'][str(rate)]['reviews']:,} | {cells} |")

    lines += ["", "## Eşlenmiş satır bootstrap aralığı", "",
              f"%5 bütçedeki TP farkı, her iki model için aynı satırlar "
              f"{gaps['gradient_boosting']['resamples']} kez yeniden örneklenerek hesaplandı. "
              "Modeller yeniden eğitilmedi. Bu aralık mevcut döneme ve eğitilmiş modellere koşulludur. "
              "Satırların bağımsız olduğu varsayımı entity ve zaman bağımlılığını korumaz; aralığın nominal "
              "%95 kapsamı garanti değildir. Hatanın yönü ve büyüklüğü burada ölçülmedi. "
              "Sıfırı dışlaması, farkın gelecek dönemlerde korunacağını göstermez.", "",
              "| Model | Gözlenen fark | %95 aralık | Sıfırı dışlıyor mu |", "|---|---:|---:|---|"]
    for key, gap in gaps.items():
        lines.append(f"| {NAMES[key]} | {gap['observed_gap']:+d} | [{gap['ci_low']:.0f}, {gap['ci_high']:.0f}] | "
                     f"{'evet' if gap['excludes_zero'] else 'hayır'} |")

    sens = summary["sensitivity_without_time_proxy"]
    removed = ", ".join(f"`{c}`" for c in sens["removed"])
    lines += ["", "## Zaman taşıyan feature'lar çıkarılınca", "",
              f"Çıkarılan kolonlar: {removed}. Bu kolonlar zamanın ilerlemesini taşır. "
              "Bunların kullanılması tek başına sızıntı anlamına gelmez; ablation, bu girdilere duyarlılığı ölçer. "
              "Aşağıdaki değerler aynı değerlendirme dönemine aittir:", "",
              "| Model | AP | ROC AUC | TP %1 | TP %5 | TP %10 | Tam kümeye göre |", "|---|---:|---:|---:|---:|---:|---|"]
    for key, row in sens["models"].items():
        full = summary["models"][key]
        moves = []
        if row["average_precision"] != full["average_precision"]:
            moves.append("AP " + ("yükseldi" if row["average_precision"] > full["average_precision"] else "düştü"))
        if row["roc_auc"] != full["roc_auc"]:
            moves.append("AUC " + ("yükseldi" if row["roc_auc"] > full["roc_auc"] else "düştü"))
        for rate in BUDGETS:
            delta = row["budgets"][str(rate)]["tp"] - full["budgets"][str(rate)]["tp"]
            if delta:
                moves.append(f"%{rate*100:.0f} {delta:+d}")
        cells = " | ".join(str(row["budgets"][str(r)]["tp"]) for r in BUDGETS)
        lines.append(f"| {NAMES[key]} | {row['average_precision']:.4f} | {row['roc_auc']:.4f} | {cells} | "
                     f"{', '.join(moves) or 'değişmedi'} |")
    for key, row in sens["models"].items():
        lines += ["", sensitivity_statement(NAMES[key], result[key], row)]
    lines += ["", "Bu nokta tahminleri anlamlılık veya eşdeğerlik testi değildir. Zaman bilgisi diğer geçmiş "
              "feature'larında da bulunabilir. Bu ablation sızıntıyı dışlamaz ve zaman etkisini bütünüyle "
              "izole etmez. Dönemler arası tutarlılık ayrıca kronolojik değerlendirmeyle ölçülmelidir."]

    gap = gb["roc_auc"] - raw["roc_auc"]
    gb_budget = gb["budgets"]["0.05"]["tp"]
    ratio = f" Oran {gb_budget/budget['tp']:.1f}." if budget["tp"] else " Ham skorun TP sayısı sıfır; oran hesaplanmadı."
    lines += ["", "## Yorum", "",
              f"Aynı {budget['reviews']:,} incelemede gözetimsiz skor "
              f"{budget['tp']} fraud yakalarken gradient boosting {gb_budget} yakalıyor." + ratio + f" AUC farkı {gap:.3f}.", "",
              "Girdi kümeleri farklı olduğundan bu fark yalnız etiket kullanımına atfedilemez. "
              "Ham veri kolonlarının tamamıyla ayrı bir model araması yapılmadı; bu sonuç bir performans tavanı değildir.", "",
              "Gözetimli modeller eğitim etiketlerinin hazır olduğunu varsayar. Veri etiketin ne zaman "
              "kesinleştiğini içermediğinden bu varsayımın üretimdeki karşılığı bilinmiyor. Gözetimsiz motor "
              "etiket olmadan skor üretebilir; bu, yeni fraud desenlerini yakaladığının kanıtı değildir.", "",
              "Hibrit kullanımın ek faydası bu tabloda ölçülmedi. Bunun için aynı toplam inceleme bütçesinde "
              "birleşik kararın ve yalnız gözetimli modelin karşılaştırılması gerekir.", "",
              f"Sayıların kaynağı: `{evidence_source}`. Yeniden çalıştırma: "
              "`python scripts/supervised_baseline.py`.", ""]
    (output_path or ROOT / "reports/supervised_baseline.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/baseline")
    args = parser.parse_args()
    main(args.data_root, args.output)
