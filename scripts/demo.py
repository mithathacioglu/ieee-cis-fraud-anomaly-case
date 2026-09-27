"""Üç validation işlemini API üzerinden geçirir.

Varsayılan: yerel LLM kaynak seçimi. --no-rag: hızlı, modelsiz gösterim.
Tek dosyalık HTML, tam JSON kanıtı ve kısa bir Markdown rapor yazar.
"""

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from html import escape
from pathlib import Path

from fastapi.testclient import TestClient

from fraud_case.api import create_app

ROOT = Path(__file__).resolve().parents[1]
CASES = [
    (3341325, "Alışıldık faaliyet", "monitor", "R10_familiar_activity", False,
     "Geçmiş davranışla uyumlu işlem izleme önerisi alır. Bu, güvenli işlem veya ödeme onayı demek değildir."),
    (3341328, "İnceleme gerektiren skor", "review", "R09_high_adjusted_score", False,
     "Bağlam sonrası skor kural eşiğini aştığı için inceleme önerilir. Skor bir fraud olasılığı değildir."),
    (3400481, "İki kural, tek karar", "review", "R03_daily_velocity", True,
     "Alışıldık faaliyet izlemeyi önerirken günlük işlem sıklığı incelemeyi önerir. Eşleşen kuralların önceliği çatışmayı çözer."),
]


def rule_rows(body):
    return [rule for rule in body["explanation"]["rules"]["rules"] if rule["matched"]]


def html_report(results, include_rag):
    cards = []
    for item in results:
        body, scores = item["response"], item["response"]["scores"]
        rows = "".join(f"<tr><td>{escape(r['id'])}</td><td>{r['priority']}</td><td>{escape(r['action'])}</td><td>{escape(r['status'])}</td></tr>" for r in rule_rows(body))
        layers = ""
        for layer in ("column", "multivariate", "entity", "temporal"):
            value = scores[f"{layer}_normalized"]
            label = "geçmiş yetersiz" if value is None else f"{value:.4f}"
            width = 0 if value is None else 100 * value
            layers += f'<div class="layer"><span>{layer}</span><div class="track"><i style="width:{width:.4f}%"></i></div><b>{label}</b></div>'
        trace = " -> ".join(event["recipient"] for event in body["agent_trace"])
        rag = body.get("rag")
        answer = escape(rag["answer"]) if rag else "Bu çalıştırmada LLM kullanılmadı (--no-rag)."
        sources = ", ".join(rag["citations"]) if rag else "yok"
        matched_evidence = json.dumps(rule_rows(body), indent=2, ensure_ascii=False, allow_nan=False)
        cards.append(f'''<article id="case-{item['transaction_id']}">
<p class="eyebrow">İŞLEM {item['transaction_id']}</p><h2>{escape(item['title'])}</h2>
<p>{escape(item['purpose'])}</p><div class="metrics"><div>Motor kararı<strong>{body['rule_decision']}</strong></div>
<div>Ham skor<strong>{scores['raw_anomaly_score']:.4f}</strong></div><div>Bağlam sonrası<strong>{scores['adjusted_anomaly_score']:.4f}</strong></div></div>
<h3>Dört anomali katmanı</h3>{layers}<p class="muted">Katman değerleri train referansındaki sıralardır; birleşik skor fraud olasılığı değildir.</p>
<h3>Eşleşen kurallar</h3><div class="scroll"><table><thead><tr><th>Kural</th><th>Öncelik</th><th>Öneri</th><th>Sonuç</th></tr></thead><tbody>{rows}</tbody></table></div>
<details><summary>Gözlenen değerler ve eşikler</summary><pre>{escape(matched_evidence)}</pre></details>
<h3>Görev akışı</h3><p class="trace">{escape(trace)}</p>
<h3>Kaynaklı açıklama</h3><p class="muted">Seçilen kaynaklar: {escape(sources)} · Kaynak seçimi LLM, işlem bilgileri motor çıktısı.</p><pre class="answer">{answer}</pre>
</article>''')
    mode = "Gerçek yerel LLM ile kaynak seçimi" if include_rag else "Hızlı mod · LLM çalıştırılmadı"
    return '''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fraud case · Üç işlem demosu</title><style>
*{box-sizing:border-box}
body{margin:0;background:#f3f5f2;color:#192f2b;font:16px/1.6 system-ui,sans-serif}
main{max-width:1020px;margin:auto;padding:48px 24px}
header{margin-bottom:30px}
h1{font-size:clamp(30px,5vw,48px);line-height:1.15;margin:12px 0}
h2{font-size:27px;margin:8px 0}
h3{font-size:17px;margin:26px 0 10px}
a{color:#176248}

.eyebrow{font-size:12px;letter-spacing:.14em;font-weight:700;color:#356858}
.muted{color:#576b63;font-size:14px}

nav{display:flex;gap:12px;flex-wrap:wrap;margin:25px 0}
nav a{padding:9px 14px;border:1px solid #becfc4;border-radius:8px;text-decoration:none}

article{background:white;border:1px solid #d5dfd7;border-radius:16px;padding:28px;margin:24px 0;scroll-margin-top:20px}

.metrics{display:grid;grid-template-columns:repeat(3,1fr);gap:12px;margin:24px 0}
.metrics div{background:#eef4ee;padding:16px;border-radius:10px;font-size:13px}
.metrics strong{display:block;font-size:25px}

/* Katman cubugu: etiket | dolgu | deger */
.layer{display:grid;grid-template-columns:115px 1fr 90px;align-items:center;gap:12px;margin:10px 0;font-size:14px}
.layer b{text-align:right;font-size:13px}
.track{height:8px;background:#e6ece5;border-radius:6px;overflow:hidden}
.track i{display:block;background:#47775c;height:100%}

table{border-collapse:collapse;width:100%;font-size:14px;text-align:left}
th,td{padding:10px;border-bottom:1px solid #e3e8e2}
.scroll{overflow:auto}

details{margin-top:18px}
summary{cursor:pointer;color:#176248}
pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.65 ui-monospace,monospace;
    background:#f4f6f2;padding:16px;border-radius:8px;max-height:420px;overflow:auto}
.answer{font:14px/1.7 system-ui,sans-serif}
.trace{font-size:13px;overflow-wrap:anywhere;border-left:3px solid #47775c;padding-left:12px}
footer{font-size:14px;color:#576b63}

@media(max-width:600px){
  main{padding:24px 14px}
  article{padding:18px}
  .metrics{grid-template-columns:1fr}
  .layer{grid-template-columns:90px 1fr 80px}
}
</style><main><header><p class="eyebrow">LOGO · DATA SCIENCE CASE</p><h1>Bir işlem neden incelemeye gider?</h1>
<p>Üç gerçek validation kaydıyla skor, iş bağlamı, kural önceliği ve yerel RAG akışı.</p><p class="muted">''' + mode + ''' · Profil: baseline · Harici web kaynağı gerektirmeyen kayıtlı gösterim.</p>
<nav>''' + "".join(f'<a href="#case-{x[0]}">{escape(x[1])}</a>' for x in CASES) + '''</nav></header>''' + "".join(cards) + '''<footer>
Bu üç örnek bir başarı ölçümü değildir; fraud etiketleri okunmadan seçildi. Kararlar inceleme/izleme önerisidir.
Context her bütçede fayda göstermedi; ürün riski adayının kazanım ve kayıpları reports/product_context.md içinde birlikte raporlanır.
HTML kayıtlı API çıktısını gösterir; canlı yeniden çalıştırma: python scripts/demo.py. Tam kanıt: demo.json.</footer></main></html>'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-rag", action="store_true", help="Skip local LLM; clearly mark the output as deterministic-only")
    args = parser.parse_args()
    results = []
    with TestClient(create_app(ROOT, context_profile="behavior_context")) as client:
        for txid, title, action, winner, conflict, purpose in CASES:
            started = time.perf_counter()
            score = client.post("/score", json={"transaction_id": txid})
            rules = client.post("/rules/evaluate", json={"transaction_id": txid})
            response = client.post("/explain", json={"transaction_id": txid, "include_rag": not args.no_rag,
                "question": "Bu işlem için motorun kararı neden verildi? Eşleşen kuralları, önceliği ve sınırları açıkla."})
            for result in (score, rules, response):
                result.raise_for_status()
            body = response.json()
            assert body["rule_decision"] == score.json()["rule_decision"] == rules.json()["action"] == action
            assert body["winning_rule"] == rules.json()["winning_rule"] == winner
            assert body["explanation"]["rules"]["action_conflict"] == conflict
            assert body["scores"] == score.json()["scores"]
            if not args.no_rag:
                rag = body["rag"]
                assert rag["model_used"] and not rag["abstained"], rag
                assert rag["authoritative_evidence"]["decision"] == action
                assert rag["authoritative_evidence"]["winning_rule"] == winner
                assert rag["authoritative_evidence"]["adjusted_anomaly_score"] == body["scores"]["adjusted_anomaly_score"]
                sources = {s["id"]: s["text"] for s in rag["sources"]}
                assert all(q["text"] == sources[q["source_id"]] for q in rag["source_quotes"])
            results.append({"transaction_id": txid, "title": title, "purpose": purpose,
                            "seconds": round(time.perf_counter() - started, 3), "response": body})
            print(f"{txid}: {action} / {winner} / conflict={conflict}", flush=True)
    out = ROOT / "artifacts/demo" / ("no_rag" if args.no_rag else "full")
    out.mkdir(parents=True, exist_ok=True)
    source_hashes = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in sorted((ROOT / "src/fraud_case").glob("*.py"))}
    record = {"created_utc": datetime.now(timezone.utc).isoformat(), "llm_enabled": not args.no_rag,
              "transport": "ASGI API via TestClient; local model over real loopback HTTP",
              "source_sha256": source_hashes, "labels_read": False, "cases": results}
    (out / "demo.json").write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (out / "index.html").write_text(html_report(results, not args.no_rag), encoding="utf-8")
    if not args.no_rag:
        lines = ["# Üç işlem demosu", "", "Çalıştırma: `python scripts/demo.py`. Ollama önceden başlatılmış olmalıdır. Hızlı, LLM'siz sürüm: `python scripts/demo.py --no-rag`.", "",
                 "Gerçek validation kayıtları, fraud etiketleri okunmadan seçildi. Her kayıt `/score`, `/rules/evaluate`, "
                 "`/explain` üzerinden çalıştırılır; karar ve skor eşitliği kontrol edilir. Bu örnekler performans benchmark'ı "
                 "değildir.", "",
                 "| İşlem | Senaryo | Karar | Kazanan kural | Ham skor | Bağlam sonrası |", "|---|---|---|---|---:|---:|"]
        for item in results:
            body, scores = item["response"], item["response"]["scores"]
            lines.append(f"| {item['transaction_id']} | {item['title']} | {body['rule_decision']} | {body['winning_rule']} | {scores['raw_anomaly_score']:.6f} | {scores['adjusted_anomaly_score']:.6f} |")
        lines += ["", "## Gösterim sırası", "", "1. Alışıldık faaliyet: R10 neden monitor öneriyor; bunun güvenlik onayı olmadığını göster.",
                  "2. Yüksek skor: dört katmanı ve R09 eşiğini aç; skorun fraud olasılığı olmadığını açıkla.",
                  "3. Çatışma: R03 review ve R10 monitor birlikte eşleşir; 90 önceliği 10'a üstün gelir. Eşleşmeyen daha yüksek "
                  "öncelikli kurallar yarışmaz.", "",
                  "Her kartta gözlenen değer/eşik, eşleşen kurallar, dört ajan izi ve kaynaklı açıklama bulunur. LLM ilgili "
                  "kaynakları seçer; işlem kararı ve sayılar motor çıktısından yazılır, kaynaklar tam metin alıntıdır.", "",
                  "Yerel çıktı: `artifacts/demo/full/index.html`; tam yanıtlar ve kaynak hash'leri: "
                  "`artifacts/demo/full/demo.json`. HTML dış script/font/CDN kullanmaz. Gösterim kayıtlı çıktıdır; canlı tekrar "
                  "komutu yukarıdadır.", "",
                  "Context sonuçlarının bütçeye bağlı kazanım/kayıpları [ürün context raporunda](product_context.md), genel case "
                  "eşleştirmesi [gereksinimlerde](../docs/requirements.md).", ""]
        (ROOT / "reports/demo.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Demo ready: {out / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
