"""Metrikler. Bütçe eşitliğinde transaction id'ye göre kırılıyor ki tekrar çalıştırınca aynı çıksın."""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def validate_labels(labels, scores):
    y, s = np.asarray(labels), np.asarray(scores, dtype=float)
    if y.ndim != 1 or s.ndim != 1 or len(y) != len(s) or not len(y):
        raise ValueError("Expected equally sized nonempty one-dimensional inputs")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(s).all():
        raise ValueError("Invalid labels or scores")
    return y.astype(bool), s


def metrics(labels, scores, *, threshold=None, review_count=None, transaction_ids=None):
    y, s = validate_labels(labels, scores)
    if (threshold is None) == (review_count is None):
        raise ValueError("Choose exactly one threshold or review count")
    if threshold is not None:
        if not np.isfinite(threshold):
            raise ValueError("Threshold must be finite")
        flagged = s >= threshold
    else:
        if not isinstance(review_count, (int, np.integer)) or not 0 <= review_count <= len(y):
            raise ValueError("Invalid review count")
        ids = np.asarray(transaction_ids)
        if ids.shape != s.shape or len(np.unique(ids)) != len(ids):
            raise ValueError("Unique transaction IDs are required for budget tie-breaking")
        order = np.lexsort((ids, -s))
        flagged = np.zeros(len(y), dtype=bool)
        flagged[order[:review_count]] = True
    tp, fp = int((flagged & y).sum()), int((flagged & ~y).sum())
    fn, tn = int((~flagged & y).sum()), int((~flagged & ~y).sum())
    return {
        "rows": len(y), "positives": int(y.sum()), "alerts": int(flagged.sum()),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
        "average_precision": float(average_precision_score(y, s)) if y.any() else None,
        "roc_auc": float(roc_auc_score(y, s)) if np.unique(y).size == 2 else None,
    }


def top_k_hits(y, scores, ids, count):
    """Sabit bütçede yakalanan fraud. Skor eşitliğinde TransactionID ile kırılır.

    Girdi doğrulaması bilinçli olarak yok: bu fonksiyon bootstrap döngüsünün
    içinden on binlerce kez çağrılıyor ve orada aynı satır birden fazla
    seçildiği için ID'ler zaten tekil değil. Doğrulama çağıranın işi.
    """
    order = np.lexsort((ids, -scores))
    return int(y[order[:count]].sum())


def paired_block_intervals(y, scores, ids, times, comparisons, *, hours, resamples, min_blocks, rate, seed):
    """Bütün sıralamalar için eşlenmiş, boş olmayan saat bloklarını yeniden örnekler.

    Bloklar ilk değerlendirilen zaman damgasından başlar. Gözlenen blok sayısı
    kadar blok yerine koymayla çekilir; bloklar eşit büyüklükte olmadığı için
    satır sayısı değişir, o yüzden her çekimde ceil(satır * rate) yeniden
    hesaplanır. Sabit model aralıkları yeniden eğitim belirsizliğini ve bloklar
    arasındaki entity bağımlılığını kapsamaz.

    comparisons: {ad: (referans_siralama, aday_siralama)}. Aynı blok çekimi
    bütün sıralamalara uygulandığı için farklar eşlenmiş kalır.
    """
    if hours <= 0 or resamples < 1 or not 0 < rate <= 1:
        raise ValueError("Invalid bootstrap configuration")
    if not len(y) or len(y) != len(times) or np.any(np.diff(times) < 0):
        raise ValueError("Bootstrap rows must be nonempty and chronological")
    missing = {name for pair in comparisons.values() for name in pair} - set(scores)
    if not comparisons or missing:
        raise ValueError(f"Comparisons reference unknown rankings: {sorted(missing)}")
    labels = np.floor((times - times[0]) / (hours * 3600)).astype(np.int64)
    blocks = np.split(np.arange(len(y)), np.flatnonzero(np.diff(labels)) + 1)
    meta = {"method": "paired_nonoverlapping_time_block_percentile", "hours": hours,
            "blocks": len(blocks), "resamples": resamples, "rate": rate,
            "assumption": "Observed time blocks are exchangeable; cross-block entity dependence remains.",
            "scope": "Conditional on fitted models and this evaluation window; not a future-period guarantee."}
    if len(blocks) < min_blocks:
        return {**meta, "status": "insufficient_blocks", "comparisons": {}}
    rng = np.random.default_rng(seed)
    gaps = {name: np.empty(resamples, dtype=int) for name in comparisons}
    sample_sizes = []
    used = {name for pair in comparisons.values() for name in pair}
    for rep in range(resamples):
        chosen = rng.integers(0, len(blocks), len(blocks))
        pick = np.concatenate([blocks[b] for b in chosen])
        sample_sizes.append(len(pick))
        count = int(np.ceil(len(pick) * rate))
        hits = {name: top_k_hits(y[pick], scores[name][pick], ids[pick], count) for name in used}
        for name, (base, challenger) in comparisons.items():
            # Oran farkı, farklı büyüklükteki çekimleri gözlenen pencereyle karşılaştırılabilir tutuyor.
            gaps[name][rep] = hits[challenger] - hits[base]
    result = {}
    observed_count = int(np.ceil(len(y) * rate))
    for name, (base, challenger) in comparisons.items():
        observed = top_k_hits(y, scores[challenger], ids, observed_count) - top_k_hits(y, scores[base], ids, observed_count)
        # Aralık, orijinal pencere büyüklüğündeki bir örneğe denk TP farkı olarak yazılıyor.
        equivalent_gaps = gaps[name] / np.array(sample_sizes) * len(y)
        low, high = np.percentile(equivalent_gaps, [2.5, 97.5])
        result[name] = {"observed_tp_gap": observed, "ci_low": float(low), "ci_high": float(high),
                        "units": "TP difference per original evaluation-window row count",
                        "excludes_zero": bool(low > 0 or high < 0)}
    return {**meta, "status": "ok", "sample_rows_min": min(sample_sizes), "sample_rows_max": max(sample_sizes),
            "comparisons": result}


def paired_effect(labels, baseline, adjusted, threshold):
    y, raw = validate_labels(labels, baseline)
    _, changed = validate_labels(labels, adjusted)
    before, after = raw >= threshold, changed >= threshold
    return {
        "removed_false_positives": int((before & ~after & ~y).sum()),
        "added_false_positives": int((~before & after & ~y).sum()),
        "lost_true_positives": int((before & ~after & y).sum()),
        "added_true_positives": int((~before & after & y).sum()),
    }
