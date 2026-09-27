"""Ürün riski adayının kazanımının belirsizliğini ölçer.

Ürün context'i aynı %5 inceleme bütçesinde 230 yerine 242 fraud yakalıyor.
Aradaki 12 işlem, "kazandı" demek için yeterli mi bilmiyordum; ölçmedikçe de
bilinemez. Burada kronolojik deneyde kullandığım eşlenmiş blok bootstrap'ının
aynısını bu karşılaştırmaya uyguluyorum ve farkı audit döneminin iki yarısında
ayrı ayrı gösteriyorum.

Neyi ölçmediğimi de yazıyorum: modeller sabit, bloklar arasında aynı entity'nin
bağımlılığı korunmuyor, bu segment context geliştirmesi sırasında zaten
görüldü. Yani sonuç "gelecek dönemde de böyle olacak" demiyor; bu pencerede
farkın sıfırdan ayrılıp ayrılmadığını söylüyor.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_case.evaluate_context import validation_sections
from fraud_case.evaluation import paired_block_intervals, top_k_hits

ROOT = Path(__file__).resolve().parents[1]
COMPARISONS = {
    "behavior_minus_raw": ("raw", "behavior_context"),
    "product_minus_raw": ("raw", "product_context"),
    "product_minus_behavior": ("behavior_context", "product_context"),
}


def load_segment():
    """Ürün deneyinin raporladığı audit yarısını aynı fonksiyonla kurar."""
    fraction = json.loads((ROOT / "config/context_evaluation.json").read_text(encoding="utf-8"))["validation_calibration_fraction"]
    filters = [("split", "=", "validation")]
    source = pd.read_parquet(ROOT / "data/processed/transactions.parquet", filters=filters,
                             columns=["TransactionID", "TransactionDT", "isFraud"]).reset_index(drop=True)
    raw = pd.read_parquet(ROOT / "data/processed/scores.parquet", filters=filters,
                          columns=["TransactionID", "raw_anomaly_score"]).reset_index(drop=True)
    behavior = pd.read_parquet(ROOT / "data/processed/context_scores.parquet", filters=filters,
                               columns=["TransactionID", "adjusted_anomaly_score"]).reset_index(drop=True)
    product = pd.read_parquet(ROOT / "artifacts/product_context/validation_scores.parquet").set_index("TransactionID")
    if not source.TransactionID.equals(raw.TransactionID) or not source.TransactionID.equals(behavior.TransactionID):
        raise ValueError("Validation score files are not aligned with transactions")
    if source.TransactionID.duplicated().any():
        raise ValueError("Validation transaction IDs must be unique")
    frame = source.copy()
    frame["raw"] = raw.raw_anomaly_score
    frame["behavior_context"] = behavior.adjusted_anomaly_score
    frame["product_context"] = frame.TransactionID.map(product.adjusted_anomaly_score)
    if frame.product_context.isna().any():
        raise ValueError("Product-context scores do not cover the validation segment")
    _, audit_mask, boundary = validation_sections(frame, fraction)
    return frame.loc[audit_mask].reset_index(drop=True), float(boundary)


def observed_gaps(frame, rates):
    y = frame.isFraud.to_numpy()
    ids = frame.TransactionID.to_numpy()
    rows = {}
    for rate in rates:
        count = int(np.ceil(len(frame) * rate))
        hits = {name: top_k_hits(y, frame[name].to_numpy(), ids, count)
                for name in ("raw", "behavior_context", "product_context")}
        rows[str(rate)] = {"reviews": count, "tp": hits,
                           "gaps": {name: hits[challenger]-hits[base] for name, (base, challenger) in COMPARISONS.items()}}
    return rows


def main():
    config = json.loads((ROOT / "config/temporal_validation.json").read_text(encoding="utf-8"))
    frame, boundary = load_segment()
    y = frame.isFraud.to_numpy()
    ids = frame.TransactionID.to_numpy()
    times = frame.TransactionDT.to_numpy(dtype=float)
    scores = {name: frame[name].to_numpy(dtype=float) for name in ("raw", "behavior_context", "product_context")}
    intervals = {}
    for hours in config["block_hours"]:
        for rate in config["review_rates"]:
            key = f"{hours}h_at_{rate}"
            intervals[key] = paired_block_intervals(y, scores, ids, times, COMPARISONS, hours=hours,
                                                    resamples=config["bootstrap_resamples"],
                                                    min_blocks=config["min_blocks"], rate=rate, seed=config["seed"])
    # Donem tutarliligi: ayni aritmetik audit yarisinin iki kronolojik parcasinda.
    first_mask, second_mask, split_boundary = validation_sections(frame, .5)
    halves = {"first_half": observed_gaps(frame.loc[first_mask], config["review_rates"]),
              "second_half": observed_gaps(frame.loc[second_mask], config["review_rates"])}
    primary = str(config["primary_review_rate"])
    consistent = {name: [halves[half][primary]["gaps"][name] for half in halves] for name in COMPARISONS}
    summary = {
        "scope": "Later-validation audit segment, already seen during context development. Not an independent holdout.",
        "segment_rows": len(frame), "segment_positives": int(y.sum()),
        "validation_audit_boundary_seconds": boundary, "half_boundary_seconds": float(split_boundary),
        "protocol": {key: config[key] for key in ("block_hours", "bootstrap_resamples", "min_blocks", "review_rates",
                                                  "primary_review_rate", "seed")},
        "comparisons": {name: list(pair) for name, pair in COMPARISONS.items()},
        "observed": observed_gaps(frame, config["review_rates"]),
        "block_bootstrap": intervals,
        "chronological_halves": halves,
        "primary_gap_sign_consistent": {name: bool(all(v > 0 for v in values) or all(v < 0 for v in values))
                                        for name, values in consistent.items()},
        "limitations": [
            "Models and the product policy are fixed; refitting uncertainty is not included.",
            "Blocks are assumed exchangeable; the same card/address entity can appear in several blocks.",
            "This segment was already used while developing the behavioural context, so it is a development check.",
            "An interval that excludes zero in this window is not evidence for a later period.",
        ],
        "test_labels_used": False,
        "input_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                         ("artifacts/product_context/validation_scores.parquet", "scripts/evaluate_product_uncertainty.py",
                          "src/fraud_case/evaluation.py")},
    }
    folder = ROOT / "artifacts/product_uncertainty"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render(summary)
    print(json.dumps({"observed_primary": summary["observed"][primary],
                      "primary_intervals": {h: intervals[f"{h}h_at_{config['primary_review_rate']}"]["comparisons"]
                                            for h in config["block_hours"]},
                      "halves_primary": {half: halves[half][primary]["gaps"] for half in halves}}, indent=2), flush=True)


def direction(first, second):
    """Iki donemdeki farkin yonu. Sifir 'ters yon' degil, 'fark yok'."""
    if first == 0 or second == 0:
        return "biri sıfır"
    return "aynı" if (first > 0) == (second > 0) else "ters"


# Bir butcedeki kanitin alabilecegi durumlar. Ayri ayri yazmamin nedeni:
# "hesaplanamadi" ile "hesaplandi, sifiri kapsiyor" ayni cumleye dusmemeli.
# Bu ayrimi skorlama katmaninda zaten uyguluyorum (eksik anomali katmani null
# doner, sifir donmez); rapor katmaninda da ayni kurali istiyorum.
READINGS = {
    "gain": "bu pencerede kazanım var",
    "loss": "bu pencerede zarar var",
    "flat": "fark gözlenmedi",
    "weak": "yön tutarlı ama belirsizlik sıfırı kapsıyor",
    "conflict": "kanıtlar çelişiyor",
    "none": "kanıt yok",
    "partial": "kanıt eksik: blok uzunluklarının bir kısmı hesaplanamadı",
    "uncomputed": "aralık hiç hesaplanamadı",
}


def interval_sign(low, high):
    """Aralik tamamen sifirin ustunde mi (+1), altinda mi (-1), sifiri kapsiyor mu (0)."""
    if low > 0:
        return 1
    if high < 0:
        return -1
    return 0


def block_evidence(summary, rate, comparison="product_minus_raw"):
    """Her blok uzunlugu icin hesaplanip hesaplanmadigi, aralik ve isareti.

    `excludes_zero` tek basina yetmiyor: aralik sifiri disliyor olabilir ama
    nokta tahmininin ters yonunde olabilir. O yuzden sinirlari ve isareti de
    tasiyorum. Yetersiz blokta `computed` False kaliyor; onceden bu bloklar
    listeden sessizce dusuyordu, yarim kanit tam kanit gibi okunabiliyordu.
    """
    states = []
    for hours in summary["protocol"]["block_hours"]:
        block = summary["block_bootstrap"][f"{hours}h_at_{rate}"]
        if block["status"] != "ok":
            states.append({"hours": hours, "computed": False, "sign": None, "ci_low": None, "ci_high": None})
            continue
        gap = block["comparisons"][comparison]
        states.append({"hours": hours, "computed": True, "ci_low": gap["ci_low"], "ci_high": gap["ci_high"],
                       "sign": interval_sign(gap["ci_low"], gap["ci_high"])})
    return states


def reading(gap, states, first, second):
    """Bir butcedeki kanitin okunmasi. READINGS anahtari dondururur.

    Bes soru ayri ayri sorulmak zorunda ve hicbiri digerinin yerine gecmiyor:

      1. Aralik hesaplanabildi mi? Hicbiri hesaplanmadiysa "belirsizlik sifiri
         kapsiyor" demek yanlis gerekce olur.
      2. Blok uzunluklarinin hepsi hesaplandi mi? Yarim kanit tam kanit degil.
      3. Aralik sifiri disliyor mu?
      4. Araligin YONU nokta tahminiyle ayni mi? Sifiri dislayan bir aralik
         nokta tahmininin tersi yonde olabilir; o durumda kazanim da zarar da
         yazilamaz. Iki blok uzunlugu birbirinin tersini soyluyorsa da oyle.
      5. Fark iki donemde ayni yonde mi? Ve fark sifirsa bu "kanit yok" degil,
         "fark gozlenmedi" demektir.
    """
    computed = [state for state in states if state["computed"]]
    if not states or not computed:
        return "uncomputed"
    if len(computed) < len(states):
        return "partial"
    signs = {state["sign"] for state in computed}
    if 1 in signs and -1 in signs:
        # Bir blok uzunlugu pozitif, digeri negatif diyor.
        return "conflict"
    excludes = 0 not in signs
    bound = next(iter(signs)) if excludes else 0
    if excludes and gap != 0 and (gap > 0) != (bound > 0):
        # Aralik sifiri disliyor ama nokta tahmininin ters yonunde.
        return "conflict"
    if excludes and gap == 0:
        return "conflict"
    if gap == 0:
        return "flat"
    consistent = direction(first, second) == "aynı"
    if excludes and consistent and gap > 0 and first > 0:
        return "gain"
    if excludes and consistent and gap < 0 and first < 0:
        return "loss"
    if consistent and not excludes:
        return "weak"
    return "none"


def budget_label(rate):
    return f"%{float(rate)*100:g}"


def opening(observed):
    """Giris cumlesini gozlenen farklardan uretir.

    Onceden burada "ham skordan daha fazla fraud yakaliyor" sabit yaziliydi.
    Kendi tablosunda %1 butcesi negatifken bile boyle diyordu.
    """
    gains = [r for r, row in observed.items() if row["gaps"]["product_minus_raw"] > 0]
    losses = [r for r, row in observed.items() if row["gaps"]["product_minus_raw"] < 0]
    flat = [r for r, row in observed.items() if row["gaps"]["product_minus_raw"] == 0]
    parts = []
    for rates, word in ((gains, "daha fazla"), (losses, "daha az"), (flat, "aynı sayıda")):
        if rates:
            detail = ", ".join(f"{budget_label(r)} ({observed[r]['gaps']['product_minus_raw']:+d})" for r in rates)
            parts.append(f"{detail} bütçesinde {word}")
    return ("Ürün riski adayı ham skora göre " + "; ".join(parts) + " fraud yakalıyor. "
            "Farkların yönü kapasiteye göre değiştiği için \"kazandı\" demeden önce her bütçede bu farkın ne kadar "
            "oynadığını ölçtüm. Kullandığım yöntem kronolojik deneydeki eşlenmiş blok bootstrap'ının aynısı; iki deney "
            "de `fraud_case.evaluation.paired_block_intervals` fonksiyonunu çağırıyor.")


def render(summary):
    (ROOT / "reports/product_uncertainty.md").write_text(report_lines(summary), encoding="utf-8")


def report_lines(summary):
    observed = summary["observed"]
    lines = ["# Ürün riski kazanımının belirsizliği", "", opening(observed), "",
             f"Segment: validation'ın audit yarısı, {summary['segment_rows']:,} işlem ve {summary['segment_positives']:,} "
             "fraud. Bu bölüm context geliştirmesi sırasında zaten görüldü; bağımsız holdout değil.", "",
             "## Aynı bütçede gözlenen fark", "",
             "| Bütçe | İnceleme | Ham | Davranış context | Ürün context | Ürün - Ham |", "|---|---:|---:|---:|---:|---:|"]
    for rate, row in observed.items():
        tp = row["tp"]
        lines.append(f"| %{float(rate)*100:g} | {row['reviews']:,} | {tp['raw']} | {tp['behavior_context']} | "
                     f"{tp['product_context']} | {row['gaps']['product_minus_raw']:+d} |")
    lines += ["", "## Eşlenmiş blok bootstrap", "",
              "Bloklar saat sınırına göre kesiliyor, gözlenen blok sayısı kadar blok yerine koymayla çekiliyor ve her "
              "çekimde bütçe satır sayısına göre yeniden hesaplanıyor. Aynı çekim bütün sıralamalara uygulandığı için "
              "farklar eşlenmiş kalıyor. Aralıklar yüzdelik yöntemiyle %95.", "",
              "| Blok | Bütçe | Karşılaştırma | Gözlenen | %95 aralık | Sıfırı dışlıyor |",
              "|---|---|---|---:|---|---|"]
    for key, block in summary["block_bootstrap"].items():
        hours, rate = key.split("h_at_")
        if block["status"] != "ok":
            lines.append(f"| {hours} saat | %{float(rate)*100:g} | tamamı | | yetersiz blok ({block['blocks']}) | |")
            continue
        for name, gap in block["comparisons"].items():
            lines.append(f"| {hours} saat | %{float(rate)*100:g} | {name} | {gap['observed_tp_gap']:+d} | "
                         f"[{gap['ci_low']:.1f}, {gap['ci_high']:.1f}] | {'evet' if gap['excludes_zero'] else 'hayır'} |")
    halves = summary["chronological_halves"]
    lines += ["", "## Dönem tutarlılığı", "",
              "Aynı aritmetiği audit yarısının iki kronolojik parçasında ayrı ayrı hesapladım. Tek bir pencerede "
              "sıfırdan ayrılan bir fark, dönem değiştiğinde yön değiştiriyorsa karar için yeterli değildir.", "",
              "| Bütçe | Karşılaştırma | İlk yarı | İkinci yarı | Aynı yön |", "|---|---|---:|---:|---|"]
    for rate in observed:
        for name in summary["comparisons"]:
            first, second = halves["first_half"][rate]["gaps"][name], halves["second_half"][rate]["gaps"][name]
            lines.append(f"| %{float(rate)*100:g} | {name} | {first:+d} | {second:+d} | {direction(first, second)} |")
    lines += ["", "## Karar", "",
              "Cevap bütçeden bağımsız değil, o yüzden tek cümlede vermiyorum. Ürün riski ile ham skor arasındaki fark "
              "için üç kapasitede durum şu:", "",
              "| Bütçe | Gözlenen | Hesaplanan blok | Aralıklar sıfırı dışlıyor mu | İki yarıda yön | Okuma |",
              "|---|---:|---:|---|---|---|"]
    readings = {}
    for rate in observed:
        gap = observed[rate]["gaps"]["product_minus_raw"]
        states = block_evidence(summary, rate)
        computed = [state for state in states if state["computed"]]
        first = halves["first_half"][rate]["gaps"]["product_minus_raw"]
        second = halves["second_half"][rate]["gaps"]["product_minus_raw"]
        readings[rate] = reading(gap, states, first, second)
        # Sifiri dislama sorusuna, hesaplanmamis blok varsa "evet/hayir" demiyorum.
        # Dislaniyorsa yonu de yaziyorum: nokta tahminiyle uyusmasi ayri bir soru.
        signs = {state["sign"] for state in computed}
        if not computed:
            interval_cell = "hesaplanamadı"
        elif 0 in signs:
            interval_cell = "hayır, sıfırı kapsıyor"
        elif 1 in signs and -1 in signs:
            interval_cell = "evet ama blok uzunlukları ters yönde"
        else:
            interval_cell = "evet, " + ("pozitif" if 1 in signs else "negatif") + " yönde"
        lines.append(f"| {budget_label(rate)} | {gap:+d} | {len(computed)}/{len(states)} | {interval_cell} | "
                     f"{direction(first, second)} | {READINGS[readings[rate]]} |")
    # Asagidaki yorum cumlelerini de tablodan uretiyorum. Elle yazilmis bir
    # yorum, sayilar degistiginde sessizce yanlis olur.
    def budgets(key):
        return ", ".join(budget_label(r) for r, value in readings.items() if value == key)
    templates = {
        "gain": "{} kapasitede fark sıfırdan ayrılıyor ve iki dönemde de kazanım yönünde.",
        "loss": "{} kapasitede kanıt zarar yönünde ve tutarlı; o kapasitede bu katmanı açmak için gerekçe yok.",
        "flat": "{} kapasitede fark gözlenmedi; bu, ölçemedim demek değil, ölçtüm ve fark çıkmadı demek.",
        "weak": "{} kapasitede yön tutarlı ama belirsizlik sıfırı kapsıyor; bu farka dayanarak politika değiştirmem.",
        "conflict": "{} kapasitede kanıtlar birbiriyle çelişiyor: gözlenen fark ile aralığın yönü ya da iki blok "
                    "uzunluğunun yönü uyuşmuyor. Çelişen kanıttan sonuç çıkarmıyorum.",
        "none": "{} kapasitede kanıt yok.",
        "partial": "{} kapasitede blok uzunluklarının bir kısmı hesaplanamadı; eksik kanıtla sonuç yazmıyorum.",
        "uncomputed": "{} kapasitede hiçbir blok uzunluğu için aralık hesaplanamadı, yani bu bütçe hakkında bir şey "
                      "söyleyemiyorum. Bu, farkın sıfıra yakın olduğu anlamına gelmez.",
    }
    sentences = ["\"Model iyi mi kötü mü\" sorusunun bu veriyle tek cevabı yok; cevap kapasiteye göre değişiyor."]
    sentences += [templates[key].format(budgets(key)) for key in templates if budgets(key)]
    lines += ["", " ".join(sentences), "",
              "Bu nedenle ürün riskini varsayılan yapmıyorum ve \"kazandı\" da demiyorum. Önerim: kapasite politikası "
              "sabitlenmeden bu karar verilmemeli; hangi kapasitede çalışıldığı sabitlendikten sonra sonraki adım bu "
              "katmanı görülmemiş yeni bir dönemde ölçmek.", "",
              "Bu ölçüm modeller ve politika sabitken, bu pencerede geçerli. Yeniden eğitim belirsizliğini kapsamıyor; "
              "bloklar arasında aynı kart/adres entity'sinin tekrar görünmesi bağımsızlık varsayımını zorluyor; segment "
              "zaten görülmüş geliştirme verisi. Bu üç sınır nedeniyle buradan üretim kararı çıkarmıyorum.", "",
              "Yeniden çalıştırma: `python scripts/evaluate_product_uncertainty.py`. Çıktı: "
              "`artifacts/product_uncertainty/evaluation.json`.", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
