"""Motoru train'de fit eder, üç bölümü de skorlar, scores.parquet yazar."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn

from fraud_case.aggregation import ScoreAggregator
from fraud_case.anomaly import LAYERS, MULTIVARIATE_FEATURES, AnomalyEngine


def run(root: Path) -> None:
    started = time.perf_counter()
    config = json.loads((root / "config/scoring.json").read_text(encoding="utf-8"))
    feature_path = root / "data/processed/features.parquet"
    features = pd.read_parquet(feature_path)
    if not set(features["split"].unique()) == {"train", "validation", "test"}:
        raise ValueError("Expected three chronological partitions")
    train = features.loc[features["split"].eq("train")]
    engine = AnomalyEngine(min_history=config["minimum_entity_history"], **config["isolation_forest"])
    print(f"Fitting four layers on {len(train):,} training rows, one CPU worker ...", flush=True)
    engine.fit(train)
    train_raw = engine.score(train)
    aggregator = ScoreAggregator(config["weights"]).fit(train_raw)
    outputs, splits = [], {}
    for split in ("train", "validation", "test"):
        part = features.loc[features["split"].eq(split)]
        print(f"Scoring {split}: {len(part):,} rows ...", flush=True)
        raw = train_raw if split == "train" else engine.score(part)
        scored = aggregator.transform(raw)
        contributions = scored[[f"{layer}_contribution" for layer in LAYERS]].sum(axis=1)
        np.testing.assert_allclose(contributions, scored.raw_anomaly_score, rtol=1e-12, atol=1e-12)
        if not scored.raw_anomaly_score.between(0, 1).all():
            raise AssertionError("Aggregated score outside [0,1]")
        scored.insert(0, "TransactionID", part.TransactionID)
        scored["split"] = split
        outputs.append(scored)
        splits[split] = {
            "rows": len(scored),
            "layer_coverage": {layer: float(scored[f"{layer}_available"].mean()) for layer in LAYERS},
            "score_quantiles": {str(k): float(v) for k, v in scored.raw_anomaly_score.quantile([0, 0.5, 0.9, 0.99, 1]).items()},
        }
    all_scores = pd.concat(outputs, ignore_index=True)
    if not all_scores.TransactionID.equals(features.TransactionID):
        raise AssertionError("Score IDs are misaligned")
    output = root / "artifacts/scoring"
    output.mkdir(parents=True, exist_ok=True)
    all_scores.to_parquet(root / "data/processed/scores.parquet", index=False)
    joblib.dump({"engine": engine, "aggregator": aggregator}, output / "model.joblib", compress=3)
    # Örnek validation'dan seçilsin; test'in en yüksek skorlusu olmasın
    validation = features.loc[features["split"].eq("validation")]
    candidates = validation.loc[validation.prior_transaction_count.ge(engine.min_history)]
    example = candidates.iloc[[0]] if len(candidates) else validation.iloc[[0]]
    explanation = engine.explain(example)
    scored_example = aggregator.transform(engine.score(example)).iloc[0]
    explanation["transaction_id"] = int(example.TransactionID.iloc[0])
    explanation["raw_anomaly_score"] = float(scored_example.raw_anomaly_score)
    explanation["available_weight"] = float(scored_example.available_weight)
    explanation["contributions"] = {layer: float(scored_example[f"{layer}_contribution"]) for layer in LAYERS}
    (output / "explanation.json").write_text(json.dumps(explanation, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    with feature_path.open("rb") as stream:
        fingerprint = hashlib.file_digest(stream, "sha256").hexdigest()
    summary = {"config": config, "splits": splits, "fit_rows": len(train),
               "multivariate_features": MULTIVARIATE_FEATURES,
               "references": {name: ref.__dict__ for name, ref in engine.references.items()},
               "calibration_reference_sizes": {name: len(values) for name, values in aggregator.reference.items()},
               "feature_sha256": fingerprint, "sklearn_version": sklearn.__version__,
               "elapsed_seconds": round(time.perf_counter() - started, 3),
               "interpretation": "Unsupervised anomaly ranks, not fraud probabilities. Labels were not loaded."}
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    figures = root / "reports/figures"
    figures.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), layout="constrained")
    training_scores = all_scores.loc[all_scores["split"].eq("train")]
    for layer in LAYERS:
        values = training_scores[f"{layer}_normalized"].dropna()
        axes[0].hist(values, bins=30, histtype="step", density=True, label=layer)
    axes[0].set(title="Layer scores against training references", xlabel="Normalized anomaly rank", ylabel="Density")
    axes[0].legend(frameon=False)
    axes[1].hist(training_scores.raw_anomaly_score, bins=40, color="#256580")
    axes[1].set(title="Weighted raw anomaly score (train)", xlabel="Score, not fraud probability", ylabel="Transactions")
    fig.savefig(figures / "anomaly_scores.png", dpi=150)
    plt.close(fig)
    lines = ["# Çok katmanlı anomali skorlama", "", "## Yöntem", "",
             "Model ve normalizasyon referansları yalnızca train feature'larından fit edildi. Hedef etiketler bu komutta "
             "okunmadı. Aşağıdaki skorlar fraud olasılığı veya performans sonucu değildir.", "",
             "| Katman | Hesap |", "|---|---|",
             "| Column | log(1+tutar) değerinin train medyanından mutlak uzaklığı / robust ölçek |",
             "| Multivariate | 21 açık feature, train medyanıyla imputation + missing indicator, IsolationForest negatif score_samples |",
             "| Entity | max(tutar sapması, 2 × gözlenen ilişki yeniliği); en az 5 geçmiş işlem |",
             "| Temporal | 1/24 saat velocity ve entity saat-fazı nadirliği için train referansına göre pozitif sapmaların "
             "maksimumu; en az 5 geçmiş işlem |", "",
             "Entity tutar sapmasının paydası max(geçmiş std, geçmiş ortalamanın %10'u, 1 tutar birimi). Bu taban, çok "
             "küçük/sıfır varyansın skoru patlatmasını sınırlar. İlişki yeniliği çarpanı 2 açıklanmış başlangıç sezgisidir; "
             "optimize edildiği iddia edilmez.", "",
             "Robust ölçek IQR/1.349'dur; dağılım yoğunlaşmışsa (p90-p10)/2.563, o da sıfırsa 1 kullanılır. Kesikli "
             "dağılımlarda bu fallback ayrıca model metadata'sında saklanır.", "",
             "IsolationForest: 128 ağaç, ağaç başına en fazla 1024 train örneği, seed=42, tek CPU işçisi. ID, hedef, split, "
             "ham elapsed time ve tüm-stream sayacı modele girmez. Eksik calendar alanları da allowlist'te değildir.", "",
             "## Normalizasyon ve birleştirme", "",
             "Her katman kendi train skorlarının sıralı referansına göre normalize edilir: count(train_raw < current_raw) / "
             "N. Bu strict rank yöntemi sayesinde aynı sıfır skoruna sahip kalabalık bir grup yüksek anomali yüzdeliği "
             "almaz. Train maksimumundan büyük skor 1 olur; eksik skor eksik kalır.", "",
             "Başlangıç ağırlıkları dört katmanda eşittir (%25). Henüz bir katmanın daha iyi fraud tespiti yaptığına dair "
             "ölçüm olmadığı için eşit ağırlık seçildi. Yapılandırma `config/scoring.json` dosyasındadır. Ağırlık/threshold "
             "seçimi yapılacaksa yalnızca validation kullanılmalıdır.", "",
             "Eksik katmanlar risksiz sayılmaz. Mevcut katman ağırlıkları kendi toplamlarına bölünür. Örneğin yalnızca "
             "column ve multivariate varsa etkin ağırlıkları %50/%50, available_weight=0.5 olur. Final katkılar toplamı "
             "raw_anomaly_score'a eşittir. Farklı coverage seviyelerinde eşik davranışı değerlendirme aşamasında ayrıca "
             "ölçülmelidir.", "",
             "## Kapsama", "", "| Bölüm | İşlem | Column | Multivariate | Entity | Temporal |", "|---|---:|---:|---:|---:|---:|"]
    for split, part in splits.items():
        coverage = " | ".join(f"%{part['layer_coverage'][layer] * 100:.2f}" for layer in LAYERS)
        lines.append(f"| {split} | {part['rows']:,} | {coverage} |")
    lines += ["", "![Train skor dağılımları](figures/anomaly_scores.png)", "", "## Açıklanabilirlik", "",
              "Her işlem için katman raw/normalized skorları, availability, etkin ağırlıklar ve katkılar kaydedilir. "
              "Entity/temporal alt bileşenleri ayrıca korunur. `AnomalyEngine.explain` gözlenen özellikleri, eşikleri ve "
              "referansları döndürür. IsolationForest için henüz feature attribution uygulanmadı; model skorunu "
              "gerekçelendiren uydurma bir SHAP açıklaması üretilmez.", "",
              "## Sınırlar ve sonraki kontrol", "",
              "Anomali ile fraud farklı hedeflerdir. Bu rapor ham anomali skorlarını kapsar; context etkisinin validation "
              "değerlendirmesi [context raporundadır](context_adjustment.md). Train skor dağılımı in-sample'dır ve başarı "
              "ölçüsü değildir. Feature geçmişi validation/test sırasında kronolojik ve etiketsiz güncellenir; model ve skor "
              "referansları sabit kalır.", "",
              "Yerel çıktı: `data/processed/scores.parquet`. Model: `artifacts/scoring/model.joblib`. Metadata ve açıklama "
              "örneği aynı klasördedir. Joblib yalnızca bu projede üretilmiş güvenilir yerel model dosyaları için "
              "kullanılmalıdır.", "",
              "Kaynak: [scikit-learn IsolationForest](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html).", ""]
    (root / "reports/anomaly_scores.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Saved {len(all_scores):,} scored rows in {summary['elapsed_seconds']} seconds.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    run(parser.parse_args().root.resolve())


if __name__ == "__main__":
    main()
