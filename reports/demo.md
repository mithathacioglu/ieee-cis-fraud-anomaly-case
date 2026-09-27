# Üç işlem demosu

Çalıştırma: `python scripts/demo.py`. Ollama önceden başlatılmış olmalıdır. Hızlı, LLM'siz sürüm: `python scripts/demo.py --no-rag`.

Gerçek validation kayıtları, fraud etiketleri okunmadan seçildi. Her kayıt `/score`, `/rules/evaluate`, `/explain` üzerinden çalıştırılır; karar ve skor eşitliği kontrol edilir. Bu örnekler performans benchmark'ı değildir.

| İşlem | Senaryo | Karar | Kazanan kural | Ham skor | Bağlam sonrası |
|---|---|---|---|---:|---:|
| 3341325 | Alışıldık faaliyet | monitor | R10_familiar_activity | 0.259438 | 0.259438 |
| 3341328 | İnceleme gerektiren skor | review | R09_high_adjusted_score | 0.985492 | 0.985492 |
| 3400481 | İki kural, tek karar | review | R03_daily_velocity | 0.584221 | 0.559963 |

## Gösterim sırası

1. Alışıldık faaliyet: R10 neden monitor öneriyor; bunun güvenlik onayı olmadığını göster.
2. Yüksek skor: dört katmanı ve R09 eşiğini aç; skorun fraud olasılığı olmadığını açıkla.
3. Çatışma: R03 review ve R10 monitor birlikte eşleşir; 90 önceliği 10'a üstün gelir. Eşleşmeyen daha yüksek öncelikli kurallar yarışmaz.

Her kartta gözlenen değer/eşik, eşleşen kurallar, dört ajan izi ve kaynaklı açıklama bulunur. LLM ilgili kaynakları seçer; işlem kararı ve sayılar motor çıktısından yazılır, kaynaklar tam metin alıntıdır.

Yerel çıktı: `artifacts/demo/full/index.html`; tam yanıtlar ve kaynak hash'leri: `artifacts/demo/full/demo.json`. HTML dış script/font/CDN kullanmaz. Gösterim kayıtlı çıktıdır; canlı tekrar komutu yukarıdadır.

Context sonuçlarının bütçeye bağlı kazanım/kayıpları [ürün context raporunda](product_context.md), genel case eşleştirmesi [gereksinimlerde](../docs/requirements.md).
