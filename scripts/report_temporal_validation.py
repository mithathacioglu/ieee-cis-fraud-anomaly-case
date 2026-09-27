"""Render chronological experiment results from saved fold metrics."""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_NAMES = {"raw": "Ham anomali", "full_logistic_regression": "LR / tam",
              "full_gradient_boosting": "GB / tam", "without_time_logistic_regression": "LR / iki kolon çıkarılmış",
              "without_time_gradient_boosting": "GB / iki kolon çıkarılmış", "matched_logistic_regression": "LR / eşlenmiş",
              "matched_gradient_boosting": "GB / eşlenmiş"}


def percent(value):
    return f"%{float(value)*100:g}"


def model_names(config):
    """Hibrit etiketi yapilandirmadaki kotadan gelir.

    Onceden "%50 kota" sabit yaziliydi. hybrid_supervised_share degistiginde
    tablo dogru sayiyi, etiket eski kotayi gosteriyordu.
    """
    return {**BASE_NAMES, "hybrid": f"Hibrit / {percent(config['hybrid_supervised_share'])} kota"}


def render(result, output):
    output.write_text(report_lines(result), encoding="utf-8")


def report_lines(result):
    folds = result["folds"]
    config = result["manifest"]["protocol"]
    rate = str(config["primary_review_rate"])
    # Butun yuzde etiketleri buradan tureniyor; hicbiri elle yazilmiyor.
    primary = percent(config["primary_review_rate"])
    names = model_names(config)
    lines = ["# Kronolojik model karşılaştırması", "",
             f"Train ve validation üzerinde {config['folds']} değerlendirme dönemi kullanıldı. Her dönemde modeller yeniden eğitildi. "
             "Final test bu çalışmada kullanılmadı. Bu veri bölümleri daha önce incelendiği için sonuçlar geliştirme sonuçlarıdır.", "",
             "Deney düzeni, varsayımlar ve tekrar çalıştırma komutu [protokolde](../docs/validation_protocol.md). "
             "Eşlenmiş modeller anomali motorunun kaynak feature'larını görür. "
             "Dönem başına kolon listeleri JSON çıktısında kayıtlıdır. "
             "Bu koşuda girdi sayıları: " + "; ".join(
                 f"{variant} = {sorted({len(f['feature_columns'][variant]) for f in folds})}"
                 for variant in ("full", "without_time", "matched")) + ".", "",
             f"## {primary} inceleme bütçesi", "",
             "Etiket gecikmesi, eğitimde hangi işlemlerin kullanılabildiğini belirleyen bir varsayımdır. "
             + ", ".join(str(d) for d in config["label_delay_days"]) + " gün koşuları aynı değerlendirme işlemlerini "
             "içerir; sonuçları birbirine eklenmez.", "",
             "| Gecikme (gün) | Dönem | İşlem | Fraud | İnceleme | Ham | GB eşlenmiş | GB tam | GB iki kolon çıkarılmış | Hibrit |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for fold in folds:
        models = fold["models"]
        values = [models[name]["budgets"][rate]["tp"] for name in
                  ("raw", "matched_gradient_boosting", "full_gradient_boosting", "without_time_gradient_boosting", "hybrid")]
        lines.append(f"| {fold['label_delay_days']} | {fold['fold']} | {fold['rows']:,} | {fold['positives']:,} | "
                     f"{models['raw']['budgets'][rate]['reviews']:,} | " + " | ".join(map(str, values)) + " |")
    for delay in config["label_delay_days"]:
        group = [f for f in folds if f["label_delay_days"] == delay]
        totals = {name: sum(f["models"][name]["budgets"][rate]["tp"] for f in group) for name in names}
        reviews = sum(f["models"]["raw"]["budgets"][rate]["reviews"] for f in group)
        differences = [f["models"]["matched_gradient_boosting"]["budgets"][rate]["tp"] -
                       f["models"]["raw"]["budgets"][rate]["tp"] for f in group]
        lines += ["", f"{delay} gün gecikmede toplam {reviews:,} incelemede ham skor {totals['raw']}, "
                  f"eşlenmiş GB {totals['matched_gradient_boosting']}, tam GB {totals['full_gradient_boosting']} "
                  f"fraud yakaladı. Eşlenmiş GB ile ham skor arasındaki dönemlik farklar: "
                  + ", ".join(f"{d:+d}" for d in differences) + ". "
                  f"Hibrit, eşlenmiş GB'ye göre toplam {totals['hybrid'] - totals['matched_gradient_boosting']:+d} TP farkı verdi."]
    lines += ["", "Toplamlar dönem başına ayrı uygulanan bütçelerin toplamıdır. "
              "Model ve girdi seçiminin etkileri birlikte görülür; eşlenmiş koşul dahi farkın yalnız etiketlerden "
              "kaynaklandığını kanıtlamaz. İşlem başına maliyet bilgisi olmadığı için TP farkı parasal getiriye çevrilmedi.", "",
              "## Dönem ve eğitim kapsamı", "",
              "Zamanlar veri setinin açıklanmayan başlangıcından itibaren saniyedir; takvim tarihi değildir.", "",
              "| Gecikme | Dönem | Eğitim işlemi | Eğitim bitişi | Değerlendirme başlangıcı | Değerlendirme bitişi | Fraud oranı |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for f in folds:
        lines.append(f"| {f['label_delay_days']} | {f['fold']} | {f['train_rows']:,} | {f['train_end']:.0f} | "
                     f"{f['evaluation_start']:.0f} | {f['evaluation_end']:.0f} | %{f['prevalence']*100:.2f} |")
    lines += ["", "## Bütün bütçeler ve sıralama metrikleri", "",
              "Hibrit satırı, sabit kota ile oluşturulan inceleme sırasını ölçer; bir olasılık modelinin skoru değildir.", "",
              "| Gecikme | Dönem | Model / girdi | AP | ROC AUC | "
              + " | ".join(f"TP {percent(r)}" for r in config["review_rates"]) + " |",
              "|---:|---:|---|---:|---:|" + "---:|" * len(config["review_rates"])]
    for f in folds:
        for key, m in f["models"].items():
            cells = " | ".join(str(m["budgets"][str(r)]["tp"]) for r in config["review_rates"])
            lines.append(f"| {f['label_delay_days']} | {f['fold']} | {names[key]} | "
                         f"{m['average_precision']:.4f} | {m['roc_auc']:.4f} | {cells} |")
    lines += ["", "## Zaman bloklarıyla bootstrap", "",
              f"Aralıklar {primary} bütçedeki TP farkına aittir. Örneklerin işlem sayısı değiştiği için fark önce işlem "
              "başına hesaplanıp özgün pencere büyüklüğüne ölçeklenir. Her iki model aynı örnekle değerlendirilir. "
              "Modeller bootstrap içinde yeniden eğitilmez; bloklar arası entity bağımlılığı korunmaz. "
              "Bu aralıklar gelecekteki performansın garantisi veya çoklu karşılaştırma düzeltmesi yapılmış testler değildir.", "",
              "| Gecikme | Dönem | Blok (saat) | Blok sayısı | Karşılaştırma | Gözlenen TP farkı | %95 yüzdelik aralık |",
              "|---:|---:|---:|---:|---|---:|---:|"]
    labels = {"matched_gb_minus_raw": "GB eşlenmiş − ham", "full_gb_minus_raw": "GB tam − ham",
              "without_time_gb_minus_full": "GB iki kolon çıkarılmış − tam", "matched_lr_minus_raw": "LR eşlenmiş − ham"}
    for f in folds:
        for block in f["block_bootstrap"]:
            prefix = f"| {f['label_delay_days']} | {f['fold']} | {block['hours']} | {block['blocks']} | "
            if block["status"] != "ok":
                lines.append(prefix + "Yetersiz blok | — | — |")
            for name, gap in block["comparisons"].items():
                lines.append(prefix + f"{labels[name]} | {gap['observed_tp_gap']:+d} | "
                             f"[{gap['ci_low']:.1f}, {gap['ci_high']:.1f}] |")
    lines += ["", "## Listelerin örtüşmesi", "",
              f"Her iki liste ayrı ayrı {primary} bütçeye sahiptir. Birleşimin inceleme sayısı daha yüksektir; "
              "birleşimdeki ek yakalamalar aynı bütçedeki kazanım olarak okunmamalıdır. "
              "Aynı bütçedeki hibrit sonuçları ilk tabloda yer alır.", "",
              "| Gecikme | Dönem | Ortak inceleme | Yalnız ham listedeki TP | Yalnız eşlenmiş GB listesindeki TP | Birleşim inceleme | Birleşim TP |",
              "|---:|---:|---:|---:|---:|---:|---:|"]
    for f in folds:
        o = f["overlap_raw_matched_gb"]
        lines.append(f"| {f['label_delay_days']} | {f['fold']} | {o['shared_reviews']} | {o['raw_only_tp']} | "
                     f"{o['supervised_only_tp']} | {o['union_reviews']} | {o['union_tp']} |")
    lines += ["", "## Kararın sınırı", "",
              "Bu deney, tanımlı dönemler ve kapasite varsayımı altında model karşılaştırmasıdır. "
              "Etiket gecikmesi, inceleme maliyeti ve kaçırılan fraud kaybı gerçek operasyon verisiyle doğrulanmadan "
              "üretim kararı verilemez. İncelenmiş validation üzerinde yeni bir yöntem seçildiği için sonraki "
              "doğrulama yeni bir dönemde yapılmalıdır.", "",
              "Sayısal özet: [temporal_validation_summary.json](temporal_validation_summary.json). "
              "Kaynak: `artifacts/temporal_validation/evaluation.json`. İşlem düzeyindeki tahminler aynı klasördeki "
              "`*_predictions.parquet` dosyalarında; çalıştırma protokolü ve kaynak hash'leri `protocol.json` içindedir.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "artifacts/temporal_validation/evaluation.json")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/temporal_validation.md")
    args = parser.parse_args()
    result = json.loads(args.input.read_text(encoding="utf-8"))
    render(result, args.output)
    # Keep numerical evidence in the delivery even when artifacts/ is excluded.
    args.output.with_name(args.output.stem + "_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
