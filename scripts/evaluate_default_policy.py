"""Varsayılan servis akışının kural kararlarını geliştirme verisinde ölçer.

Sorun şu: `data/processed/rule_decisions.parquet` ve ona dayanan final rapor,
davranış context'i uygulanmış skorlardan üretildi. Servisin varsayılanını `raw`
yaptıktan sonra bu iki akış aynı kararı vermek zorunda değil, çünkü R09 eşiği
`adjusted_anomaly_score` üzerinde ve R10 de aynı alanı kullanıyor. Aynı JSON
kurallarını çalıştırmak, aynı kararları üretmek anlamına gelmiyor.

Final testi yeniden açmadan bunu kapatmanın yolu, güncel akışı geliştirme
verisinde ölçmek ve final raporun hangi politikaya ait olduğunu net söylemek.
Bu betik onu yapıyor: validation bölümünde raw profilinin kural kararlarını
üretip, teslimdeki context tabanlı kararlarla aynı satırlarda karşılaştırıyor.

Etiketli ölçüm yalnızca context geliştirmesinde zaten görülmüş sonraki
validation yarısında. Final test etiketleri okunmuyor.
"""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from fraud_case.context import ContextConfig, ContextEngine
from fraud_case.evaluate_rules import decision_metrics
from fraud_case.rules import RuleEngine

ROOT = Path(__file__).resolve().parents[1]
RULE_THRESHOLD = .85


def rule_inputs(features, context_output):
    """evaluate_rules.load_inputs ile ayni birlestirme; girdi kaynagi farkli."""
    overlap = sorted(set(features.columns) & set(context_output.columns))
    for column in overlap:
        if not features[column].equals(context_output[column]):
            raise ValueError(f"Conflicting rule input column: {column}")
    return pd.concat([features, context_output.drop(columns=overlap)], axis=1)


def equal_volume_threshold(scores, target_alerts):
    """Hedeflenen alarm sayisina denk gelen esik, ve o esigin gercekte urettigi sayi.

    Tek bir esik bagli skorlarda tam kapasite garanti etmez: k'nci skora esit
    baska satirlar varsa ">= esik" k'dan fazla alarm uretir. Ornek olarak
    [0.9, 0.8, 0.8, 0.1] ve hedef 2 icin esik 0.8 secilir ama 3 alarm cikar.
    Bu yuzden iki sayiyi birlikte donduruyorum; tam k islem secmek gerekiyorsa
    yol esik degil, TransactionID kirilimli top-k.
    """
    values = np.asarray(scores, dtype=float)
    result = {"target_alerts": int(target_alerts), "threshold": None, "threshold_alerts": None}
    if not 0 < target_alerts <= len(values):
        return result
    candidate = float(np.sort(values)[::-1][int(target_alerts)-1])
    return {**result, "threshold": candidate, "threshold_alerts": int((values >= candidate).sum())}


def main():
    frozen = json.loads((ROOT / "artifacts/context/evaluation.json").read_text(encoding="utf-8"))
    config = ContextConfig(**frozen["selected_context"])
    reference = frozen["product_reference"]
    boundary = frozen["validation_boundary_seconds"]
    filters = [("split", "=", "validation")]
    features = pd.read_parquet(ROOT / "data/processed/features.parquet", filters=filters).reset_index(drop=True)
    scores = pd.read_parquet(ROOT / "data/processed/scores.parquet", filters=filters).reset_index(drop=True)
    times = pd.read_parquet(ROOT / "data/processed/transactions.parquet", filters=filters,
                            columns=["TransactionID", "TransactionDT", "ProductCD"]).reset_index(drop=True)
    stored = pd.read_parquet(ROOT / "data/processed/rule_decisions.parquet", filters=filters).reset_index(drop=True)
    if not features.TransactionID.equals(scores.TransactionID) or not features.TransactionID.equals(times.TransactionID):
        raise ValueError("Validation inputs are not aligned")
    if not features.TransactionID.equals(stored.TransactionID):
        raise ValueError("Stored rule decisions are not aligned with validation inputs")
    features["TransactionDT"], features["ProductCD"] = times.TransactionDT, times.ProductCD
    engine = RuleEngine.from_json(ROOT / "config/rules.json")
    # Servisin iki profili: raw gucu sifirlar, behavior_context secilmis gucu uygular.
    profiles = {"raw": replace(config, strength=0.), "behavior_context": config}
    outputs, decisions = {}, {}
    for name, profile in profiles.items():
        outputs[name] = ContextEngine(profile, reference).apply(features, scores)
        decisions[name] = engine.evaluate(rule_inputs(features, outputs[name]))
    # Teslimdeki kayitli kararlar behavior_context profiline ait olmali.
    reproduced = decisions["behavior_context"].rule_action.equals(stored.rule_action)
    labels = pd.read_parquet(ROOT / "data/processed/transactions.parquet", columns=["TransactionID", "isFraud"],
                             filters=[("split", "=", "validation"), ("TransactionDT", ">=", boundary)])
    audit = features.index[features.TransactionID.isin(set(labels.TransactionID))]
    if len(audit) != len(labels):
        raise ValueError("Audit label alignment mismatch")
    y = labels.set_index("TransactionID").loc[features.loc[audit, "TransactionID"], "isFraud"].to_numpy()
    measured, activity = {}, {}
    for name, frame in decisions.items():
        segment = frame.loc[audit]
        measured[name] = {
            "rule_review": decision_metrics(y, segment.rule_action.eq("review").to_numpy()),
            "score_at_0_85": decision_metrics(y, outputs[name].loc[audit, "adjusted_anomaly_score"].ge(RULE_THRESHOLD).to_numpy()),
            "action_counts": {str(k): int(v) for k, v in segment.rule_action.value_counts().items()},
            "action_conflicts": int(segment.rule_action_conflict.sum()),
        }
        activity[name] = {rule["id"]: int(segment[f"rule_match_{rule['id']}"].sum()) for rule in engine.config["rules"]}
    changed = decisions["raw"].rule_action.ne(decisions["behavior_context"].rule_action)
    audit_changed = changed.loc[audit]
    moved = audit_changed[audit_changed].index
    transitions = (decisions["behavior_context"].rule_action.loc[moved].astype(str) + " -> "
                   + decisions["raw"].rule_action.loc[moved].astype(str))
    # Etiket kullanmadan, yalnizca alarm hacmini esitleyen R09 esigi. Oneri degil,
    # kararin hangi knob'a bagli oldugunu gostermek icin.
    behavior_alerts = int(outputs["behavior_context"].loc[audit, "adjusted_anomaly_score"].ge(RULE_THRESHOLD).sum())
    equal_volume = equal_volume_threshold(outputs["raw"].loc[audit, "adjusted_anomaly_score"], behavior_alerts)
    summary = {
        "scope": "Validation split only. Labelled comparison uses the later-validation segment already seen during "
                 "context development; final test labels are not read.",
        "rule_config_sha256": engine.config_sha256, "rule_comparison_threshold": RULE_THRESHOLD,
        "validation_rows": len(features), "audit_rows": len(audit), "audit_positives": int(y.sum()),
        "stored_decisions_reproduced_by_behavior_context": bool(reproduced),
        "stored_decisions_profile": "behavior_context",
        "final_report_policy": "reports/final_evaluation.md rule metrics belong to the behaviour-context policy, not to "
                               "the current raw default.",
        "measured": measured, "rule_matches": activity,
        "validation_decisions_changed": int(changed.sum()),
        "audit_decisions_changed": int(audit_changed.sum()),
        "audit_decision_transitions": {str(k): int(v) for k, v in transitions.value_counts().items()},
        "equal_volume_r09_threshold": equal_volume,
        "equal_volume_note": "Threshold on raw scores aimed at the behaviour-context review volume on this segment. "
                             "Found from score volume only; no labels were used and R09 was not changed. A single "
                             "threshold cannot hit an exact capacity when scores tie, so threshold_alerts reports what "
                             "the threshold actually produces; exact-k selection needs the ID-tie-broken top-k instead.",
        "recommendation": "Keep R09 at 0.85 and report the higher review volume, or recalibrate it against a stated "
                          "capacity policy. I did not recalibrate it here because that is a selection step and the "
                          "capacity policy was not given.",
        "test_labels_used": False,
        "input_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in
                         ("config/rules.json", "data/processed/rule_decisions.parquet",
                          "scripts/evaluate_default_policy.py")},
    }
    folder = ROOT / "artifacts/default_policy"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "evaluation.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    render(summary)
    print(json.dumps({"stored_reproduced": summary["stored_decisions_reproduced_by_behavior_context"],
                      "audit_decisions_changed": summary["audit_decisions_changed"],
                      "transitions": summary["audit_decision_transitions"],
                      "raw_review": measured["raw"]["rule_review"], "behavior_review": measured["behavior_context"]["rule_review"]},
                     indent=2), flush=True)


def render(summary):
    raw, behavior = summary["measured"]["raw"], summary["measured"]["behavior_context"]
    lines = ["# Varsayılan politikanın kural kararları", "",
             "Kural motoru skorları değiştirmiyor ama R09 eşiği `adjusted_anomaly_score` üzerinde çalışıyor. Servisin "
             "varsayılanını `raw` yapınca bu alan ham skora eşitleniyor, yani aynı JSON kuralları aynı kararları "
             "vermiyor. Teslimdeki `rule_decisions.parquet` ve ona dayanan final rapor davranış context'i uygulanmış "
             "skorlardan üretildi. Bu raporun amacı o farkı ölçmek ve final raporun hangi politikaya ait olduğunu "
             "net söylemek.", "",
             f"Kapsam: validation bölümünün tamamı ({summary['validation_rows']:,} işlem); etiketli ölçüm context "
             f"geliştirmesinde görülmüş sonraki yarıda ({summary['audit_rows']:,} işlem, {summary['audit_positives']:,} "
             "fraud). Final test etiketleri okunmadı.", "",
             "## Kayıtlı kararların kaynağı", "",
             ("Teslimdeki kararlar `behavior_context` profiliyle birebir yeniden üretildi. Yani final raporun kural "
              "satırları bu politikaya aittir, güncel varsayılana değil."
              if summary["stored_decisions_reproduced_by_behavior_context"] else
              "Teslimdeki kararlar `behavior_context` profiliyle yeniden üretilemedi; bu durumda kayıtlı kararların "
              "kaynağı belirsizdir ve aşağıdaki karşılaştırma tek başına yeterli değildir."), "",
             f"Aynı satırlarda iki profilin kararı {summary['audit_decisions_changed']:,} işlemde farklı "
             f"({summary['validation_decisions_changed']:,} işlem validation'ın tamamında). Değişimin dağılımı:", ""]
    for transition, count in summary["audit_decision_transitions"].items():
        lines.append(f"- {transition}: {count:,} işlem")
    lines += ["", "## Geliştirme verisinde iki politikanın karar metrikleri", "",
              "| Politika | İnceleme | TP | FP | Precision | Recall |", "|---|---:|---:|---:|---:|---:|"]

    def percent(value):
        return "n/a" if value is None else f"%{100*value:.3f}"
    for name, block in (("raw (varsayılan) + kurallar", raw["rule_review"]), ("behavior_context + kurallar", behavior["rule_review"]),
                        ("raw skor >= 0.85", raw["score_at_0_85"]), ("context skor >= 0.85", behavior["score_at_0_85"])):
        lines.append(f"| {name} | {block['reviews']:,} | {block['tp']:,} | {block['fp']:,} | "
                     f"{percent(block['precision'])} | {percent(block['recall'])} |")
    extra = raw["rule_review"]["reviews"] - behavior["rule_review"]["reviews"]
    caught = raw["rule_review"]["tp"] - behavior["rule_review"]["tp"]
    lines += ["", f"Varsayılanı `raw` yapmak bu segmentte inceleme sayısını {extra:+,} değiştiriyor ve yakalanan fraud'u "
              f"{caught:+,}. Bunu iyileşme diye sunmuyorum: context indirimi kaldırıldığı için daha çok işlem 0.85 "
              "eşiğini geçiyor, yani iki politika eşit sayıda işlem incelemeye göndermiyor. Aradaki farkın bir kısmı "
              "sıralama değil, hacim.", ""]
    volume = summary["equal_volume_r09_threshold"]
    if volume["threshold"] is not None:
        exact = volume["threshold_alerts"] == volume["target_alerts"]
        lines += [f"Hangi knob'a bağlı olduğunu göstermek için: ham skorlarda {volume['target_alerts']:,} incelemeye "
                  f"denk gelen eşik {volume['threshold']:.6f}. Bu sayı yalnızca alarm hacmi üzerinden bulundu, etiket "
                  "kullanılmadı ve R09 değiştirilmedi.", "",
                  f"Bu eşik gerçekte {volume['threshold_alerts']:,} alarm üretiyor" +
                  (", yani bu segmentte hedefi tam tutuyor. Bunu genel bir garanti olarak yazmıyorum. " if exact else
                   f", hedeflenen {volume['target_alerts']:,} değil. ") +
                  "Tek bir eşik, eşit skorlu satırlar olduğunda tam kapasite garanti etmez; k'ncı skora "
                  "eşit başka satırlar varsa hepsi eşiği geçer. Tam olarak k işlem seçmek gerekiyorsa yol eşik "
                  "değil, projenin her yerinde kullandığı TransactionID kırılımlı top-k seçimidir. Eşiği bu yüzden "
                  "bir öneri olarak değil, kararın hangi knob'a bağlı olduğunu göstermek için yazıyorum.", "",
                  "Kapasite politikası verilmediği için R09'u yeniden kalibre etmedim; bu bir seçim adımı olur ve neye "
                  "göre seçtiğimi yazamazdım.", ""]
    lines += ["## Politikaların durumu", "",
              "| Politika | Nasıl ölçüldü | Serviste yeri |", "|---|---|---|",
              "| Ham skor + kurallar | Bu raporda, geliştirme verisinde | Varsayılan (`raw`) |",
              "| Davranış context'i + kurallar | Final raporda ve kural raporunda | İsteğe bağlı (`behavior_context`) |",
              "| Ürün riski + kurallar | Geliştirme adayı, belirsizliği ölçüldü | Önerilmiyor (`product_risk`) |", "",
              "Final rapordaki kural satırları ikinci satıra aittir. Final testi yeniden açmadım, bu yüzden varsayılan "
              "politikanın final dönemindeki kural performansını bilmiyorum ve bilmediğimi yazıyorum. Ölçebildiğim şey "
              "geliştirme verisindeki fark.", "",
              "Yeniden çalıştırma: `python scripts/evaluate_default_policy.py`. Çıktı: "
              "`artifacts/default_policy/evaluation.json`.", ""]
    (ROOT / "reports/default_policy.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
