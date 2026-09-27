"""Dört anomali katmanının aynı bütçede ne kattığını ölçer.

Bu soruyu her değerlendirmede sordular ve hiç cevaplamamıştım: dört katman
gerçekten dört ayrı yararlı sinyal mi, yoksa aynı bilgiyi tekrar mı ediyorlar?
Tutar, geçmiş ve velocity birden fazla katmanda görünüyor, dolayısıyla soru
yerinde.

Ölçüm aynı inceleme kapasitesinde yakalanan fraud üzerinden. İki koşu var:

  - her katman tek başına sıralama yapsa ne yakalar
  - bir katmanı çıkarıp kalan üçünü aynı şekilde birleştirsem ne kaybederim

Belirsizlik için kronolojik deneydeki eşlenmiş blok bootstrap'ının aynısı
kullanılıyor ve kanıt okuması `evaluate_product_uncertainty` ile aynı
fonksiyondan geliyor; iki deneyin "kazanım / zarar / hesaplanamadı" ayrımı
farklı olmasın.

Ağırlıkları bu sonuca göre değiştirmiyorum. Bu görülmüş validation üzerinde bir
geliştirme ölçümü; buradan seçim yapmak, aynı veriye ikinci kez uymak olur.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from evaluate_product_uncertainty import READINGS, block_evidence, reading

from fraud_case.anomaly import LAYERS
from fraud_case.evaluate_context import validation_sections
from fraud_case.evaluation import paired_block_intervals, top_k_hits

ROOT = Path(__file__).resolve().parents[1]
# Skorlanamayan satir icin gecerli bir skor yok. Sifir vermek "risk yok" demek
# olurdu; bu projede eksik katman sifir degil null. Siralamada en sona atiyorum.
UNSCORED = -1.


def combine(frame, layers):
    """Verilen katmanlari esit agirlikla birlestirir: mevcut olanlarin ortalamasi.

    Servisteki aggregator da bunu yapiyor, yalnizca mevcut katmanlarin agirligini
    yeniden normalize ediyor. Esit agirlikta bu, mevcut normalized skorlarin
    ortalamasina esit. Dortunun ortalamasinin kayitli raw skoru birebir yeniden
    urettigini asagida dogruluyorum; uretmezse leave-one-out da guvenilir olmaz.
    """
    values = frame[[f"{layer}_normalized" for layer in layers]].to_numpy(dtype=float)
    available = np.isfinite(values)
    total = np.where(available, values, 0.).sum(axis=1)
    count = available.sum(axis=1)
    return np.where(count > 0, total / np.maximum(count, 1), UNSCORED), count


def rankings(frame):
    """raw, tek katman ve birini-cikar siralamalari."""
    scores, unscored = {}, {}
    baseline, count = combine(frame, LAYERS)
    scores["raw"] = baseline
    unscored["raw"] = int((count == 0).sum())
    for layer in LAYERS:
        values, single = combine(frame, [layer])
        scores[f"only_{layer}"] = values
        unscored[f"only_{layer}"] = int((single == 0).sum())
        remaining = [name for name in LAYERS if name != layer]
        values, without = combine(frame, remaining)
        scores[f"without_{layer}"] = values
        unscored[f"without_{layer}"] = int((without == 0).sum())
    return scores, unscored


def main():
    protocol = json.loads((ROOT / "config/temporal_validation.json").read_text(encoding="utf-8"))
    fraction = json.loads((ROOT / "config/context_evaluation.json").read_text(encoding="utf-8"))["validation_calibration_fraction"]
    filters = [("split", "=", "validation")]
    source = pd.read_parquet(ROOT / "data/processed/transactions.parquet", filters=filters,
                             columns=["TransactionID", "TransactionDT", "isFraud"]).reset_index(drop=True)
    scores = pd.read_parquet(ROOT / "data/processed/scores.parquet", filters=filters).reset_index(drop=True)
    if not source.TransactionID.equals(scores.TransactionID):
        raise ValueError("Validation scores are not aligned with transactions")
    frame = pd.concat([source, scores.drop(columns=["TransactionID", "split"])], axis=1)
    _, audit_mask, boundary = validation_sections(frame, fraction)
    audit = frame.loc[audit_mask].reset_index(drop=True)
    variants, unscored = rankings(audit)
    # Esit agirlikli birlesim kayitli raw skoru yeniden uretmeli. Uretmiyorsa
    # leave-one-out hesabim servisin yaptigi seyi olcmuyor demektir.
    scored = variants["raw"] != UNSCORED
    reproduction = float(np.abs(variants["raw"][scored] - audit.raw_anomaly_score.to_numpy()[scored]).max())
    if not reproduction < 1e-9:
        raise ValueError(f"Equal-weight recombination does not reproduce the stored score: {reproduction}")
    y = audit.isFraud.to_numpy()
    ids = audit.TransactionID.to_numpy()
    times = audit.TransactionDT.to_numpy(dtype=float)
    comparisons = {f"{name}_minus_raw": ("raw", name) for name in variants if name != "raw"}
    observed = {}
    for rate in protocol["review_rates"]:
        count = int(np.ceil(len(audit) * rate))
        hits = {name: top_k_hits(y, values, ids, count) for name, values in variants.items()}
        observed[str(rate)] = {"reviews": count, "tp": hits,
                               "gaps": {key: hits[challenger]-hits[base] for key, (base, challenger) in comparisons.items()}}
    intervals = {}
    for hours in protocol["block_hours"]:
        for rate in protocol["review_rates"]:
            intervals[f"{hours}h_at_{rate}"] = paired_block_intervals(
                y, variants, ids, times, comparisons, hours=hours, resamples=protocol["bootstrap_resamples"],
                min_blocks=protocol["min_blocks"], rate=rate, seed=protocol["seed"])
    first_mask, second_mask, split_boundary = validation_sections(audit, .5)
    halves = {}
    for label, mask in (("first_half", first_mask), ("second_half", second_mask)):
        part = audit.loc[mask]
        index = part.index.to_numpy()
        count = int(np.ceil(len(part) * protocol["primary_review_rate"]))
        hits = {name: top_k_hits(y[index], values[index], ids[index], count) for name, values in variants.items()}
        halves[label] = {"reviews": count,
                         "gaps": {key: hits[challenger]-hits[base] for key, (base, challenger) in comparisons.items()}}
    summary = {
        "scope": "Later-validation audit segment, already seen during context development. Development measurement; "
                 "weights are not re-selected from this result.",
        "segment_rows": len(audit), "segment_positives": int(y.sum()),
        "validation_audit_boundary_seconds": float(boundary), "half_boundary_seconds": float(split_boundary),
        "protocol": {key: protocol[key] for key in ("block_hours", "bootstrap_resamples", "min_blocks",
                                                    "review_rates", "primary_review_rate", "seed")},
        "layers": list(LAYERS), "comparisons": {k: list(v) for k, v in comparisons.items()},
        "equal_weight_reproduces_stored_score": True, "max_reproduction_difference": reproduction,
        "unscorable_rows": unscored,
        "unscorable_note": f"A row with no available layer cannot be ranked by that variant; it is placed last "
                           f"({UNSCORED}). Treating it as zero risk would repeat the mistake the scoring layer avoids.",
        "observed": observed, "block_bootstrap": intervals, "chronological_halves": halves,
        "limitations": [
            "Models and normalisation references are fixed; refitting uncertainty is not included.",
            "Blocks are assumed exchangeable; the same entity can appear in several blocks.",
            "This segment was already used during context development, so it is not an independent holdout.",
            "Layer inputs overlap by construction, so a single-layer result is not an independent signal test.",
        ],
        "weights_changed": False, "test_labels_used": False,
        "input_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                         ("data/processed/scores.parquet", "scripts/evaluate_layer_contribution.py",
                          "src/fraud_case/evaluation.py")},
    }
    folder = ROOT / "artifacts/layer_contribution"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (ROOT / "reports/layer_contribution.md").write_text(report_lines(summary), encoding="utf-8")
    primary = str(protocol["primary_review_rate"])
    print(json.dumps({"reproduction_difference": reproduction, "primary_budget": primary,
                      "tp": observed[primary]["tp"], "gaps": observed[primary]["gaps"]}, indent=2), flush=True)


def label(name):
    if name == "raw":
        return "Dördü birlikte"
    kind, layer = name.split("_", 1)
    return f"Yalnız {layer}" if kind == "only" else f"{layer} çıkarılmış"


def verdicts(summary, rate):
    """Her varyant icin kanit okumasi; urun belirsizligiyle ayni fonksiyon."""
    result = {}
    for key in summary["comparisons"]:
        states = block_evidence(summary, rate, comparison=key)
        gap = summary["observed"][rate]["gaps"][key]
        first = summary["chronological_halves"]["first_half"]["gaps"][key]
        second = summary["chronological_halves"]["second_half"]["gaps"][key]
        result[key] = reading(gap, states, first, second)
    return result


def layer_of(key):
    return key.replace("without_", "").replace("only_", "").replace("_minus_raw", "")


def interpretation(summary, readings, rate):
    """Okuma cumlelerini kovalardan uretir.

    Kovalar ayri tutulmak zorunda. "Yon tutarli ama belirsizlik sifiri kapsiyor"
    negatif bir farkla geldiginde bu "kayip vermiyor" DEGIL, "isaret kayip
    yonunde ama kanit sifirdan ayrilmiyor" demektir. Ilk yazdigimda ikisini ayni
    kovaya atmistim ve -29'luk tutarli bir farki "kayip vermeyen" diye
    listeliyordu.
    """
    gaps = summary["observed"][rate]["gaps"]

    def pick(prefix, predicate):
        return [layer_of(key) for key, value in readings.items() if key.startswith(prefix) and predicate(value, gaps[key])]
    established = pick("without_", lambda state, gap: state == "loss")
    leaning = pick("without_", lambda state, gap: state == "weak" and gap < 0)
    quiet = pick("without_", lambda state, gap: state in ("flat", "none") or (state == "weak" and gap > 0))
    conflicting = pick("without_", lambda state, gap: state in ("conflict", "partial", "uncomputed"))
    single_gain = pick("only_", lambda state, gap: state == "gain")
    single_loss = pick("only_", lambda state, gap: state == "loss")
    sentences = []
    if established:
        sentences.append("Çıkarıldığında ölçülebilir kayıp veren katmanlar: " + ", ".join(established)
                         + ". Bu katmanlar birleşime kendi başına bir şey katıyor.")
    if leaning:
        sentences.append("Çıkarıldığında işaret kayıp yönünde olan ama aralığı sıfırı kapsayan katmanlar: "
                         + ", ".join(leaning) + ". Bunları \"kayıp vermiyor\" diye yazmam; fark iki dönemde de aynı "
                         "yönde, yalnızca sıfırdan ayrılmıyor.")
    if quiet:
        sentences.append("Çıkarıldığında bu pencerede kayıp işareti bile görülmeyen katmanlar: " + ", ".join(quiet)
                         + ". Bu, o katmanların gereksiz olduğunu kanıtlamıyor; bu kapasitede ve bu dönemde farkın "
                         "sıfırdan ayrılmadığını söylüyor.")
    if conflicting:
        sentences.append("Kanıtı çelişen veya hesaplanamayan katmanlar: " + ", ".join(conflicting)
                         + ". Bunlar hakkında bir şey söylemiyorum.")
    if single_gain:
        sentences.append("Tek başına dördünün birleşimini geçen katmanlar: " + ", ".join(single_gain)
                         + ". Eşit ağırlığın bu kapasitede en iyi seçim olmadığını gösteren bir işaret.")
    if single_loss:
        sentences.append("Tek başına birleşimden belirgin biçimde kötü olan katmanlar: " + ", ".join(single_loss)
                         + ". Bu beklenen sonuç: tek katman daha az bilgi görüyor.")
    largest = max(gaps, key=lambda key: abs(gaps[key]))
    sentences.append(f"En büyük gözlenen etki {layer_of(largest)} tarafında ({gaps[largest]:+d}) ve okuması "
                     f"\"{READINGS[readings[largest]]}\". Gözlenen en büyük sayının kanıt olarak en güçlü sayı "
                     "olmadığını ayrıca yazıyorum, çünkü tabloya bakan önce ona bakar.")
    return sentences


def report_lines(summary):
    primary = str(summary["protocol"]["primary_review_rate"])
    observed = summary["observed"]
    budgets = [str(r) for r in summary["protocol"]["review_rates"]]
    lines = ["# Katmanların aynı bütçede katkısı", "",
             "Dört anomali katmanının dört ayrı yararlı sinyal olup olmadığı bu teslimde ölçülmemiş bir soruydu. "
             "Tutar, geçmiş ve velocity birden fazla katmanda görünüyor, dolayısıyla eşit ağırlıklı birleşimin "
             "gerçekten dört bağımsız kanıt topladığı kendiliğinden doğru değil. Burada iki şeyi ölçüyorum: her "
             "katman tek başına aynı kapasitede ne yakalıyor, ve bir katmanı çıkarınca ne kaybediyorum.", "",
             f"Segment: validation'ın audit yarısı, {summary['segment_rows']:,} işlem, "
             f"{summary['segment_positives']:,} fraud. Görülmüş geliştirme verisi; bağımsız holdout değil.", "",
             "Eşit ağırlıklı yeniden birleştirme kayıtlı raw skoru birebir üretiyor (en büyük fark "
             f"{summary['max_reproduction_difference']:.1e}), yani birini-çıkar hesabı servisin yaptığı işlemin "
             "aynısını kullanıyor. Hiçbir katmanı olmayan satır o varyantla sıralanamaz ve en sona konur; sıfır risk "
             "sayılmaz.", "",
             "## Aynı kapasitede yakalanan fraud", "",
             "| Sıralama | " + " | ".join(f"TP {budget_label(r)}" for r in budgets) + " | Sıralanamayan satır |",
             "|---|" + "---:|" * (len(budgets) + 1)]
    for name in ["raw"] + [n for n in summary["observed"][primary]["tp"] if n != "raw"]:
        cells = " | ".join(str(observed[r]["tp"][name]) for r in budgets)
        lines.append(f"| {label(name)} | {cells} | {summary['unscorable_rows'][name]:,} |")
    lines += ["", "İnceleme sayıları: " + ", ".join(f"{budget_label(r)} = {observed[r]['reviews']:,}" for r in budgets)
              + ".", "",
              "## Birincil bütçede fark ve belirsizlik", "",
              f"Aşağıdaki aralıklar {budget_label(primary)} kapasitedeki TP farkına ait, eşlenmiş blok bootstrap ile. "
              "Kanıt okuması ürün riski raporundaki fonksiyonun aynısı: aralığın hesaplanıp hesaplanmadığı, sıfırı "
              "dışlayıp dışlamadığı, yönünün gözlenen farkla uyuşup uyuşmadığı ve iki dönemde aynı yönde olup "
              "olmadığı ayrı ayrı soruluyor.", "",
              "| Karşılaştırma | Gözlenen | İlk yarı | İkinci yarı | Okuma |", "|---|---:|---:|---:|---|"]
    readings = verdicts(summary, primary)
    for key, value in readings.items():
        gap = observed[primary]["gaps"][key]
        first = summary["chronological_halves"]["first_half"]["gaps"][key]
        second = summary["chronological_halves"]["second_half"]["gaps"][key]
        lines.append(f"| {label(key.replace('_minus_raw', ''))} - dördü birlikte | {gap:+d} | {first:+d} | "
                     f"{second:+d} | {READINGS[value]} |")
    lines += ["", "## Okuma", ""] + interpretation(summary, readings, primary)
    lines += ["", "Ağırlıkları bu sonuca göre değiştirmedim ve değiştirmeyi önermiyorum. Bu segment context "
              "geliştirmesinde zaten görüldü; buradan ağırlık seçmek aynı veriye ikinci kez uymak olur. Katman "
              "girdileri tasarım gereği örtüşüyor, dolayısıyla tek katman sonucu bir bağımsızlık testi değil. "
              "Sonraki adım, bu ölçümü görülmemiş bir dönemde tekrarlamak olurdu.", "",
              "Yeniden çalıştırma: `python scripts/evaluate_layer_contribution.py`. Çıktı: "
              "`artifacts/layer_contribution/evaluation.json`.", ""]
    return "\n".join(lines)


def budget_label(rate):
    return f"%{float(rate)*100:g}"


if __name__ == "__main__":
    main()
