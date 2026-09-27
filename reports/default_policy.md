# Varsayılan politikanın kural kararları

Kural motoru skorları değiştirmiyor ama R09 eşiği `adjusted_anomaly_score` üzerinde çalışıyor. Servisin varsayılanını `raw` yapınca bu alan ham skora eşitleniyor, yani aynı JSON kuralları aynı kararları vermiyor. Teslimdeki `rule_decisions.parquet` ve ona dayanan final rapor davranış context'i uygulanmış skorlardan üretildi. Bu raporun amacı o farkı ölçmek ve final raporun hangi politikaya ait olduğunu net söylemek.

Kapsam: validation bölümünün tamamı (118,108 işlem); etiketli ölçüm context geliştirmesinde görülmüş sonraki yarıda (59,054 işlem, 2,061 fraud). Final test etiketleri okunmadı.

## Kayıtlı kararların kaynağı

Teslimdeki kararlar `behavior_context` profiliyle birebir yeniden üretildi. Yani final raporun kural satırları bu politikaya aittir, güncel varsayılana değil.

Aynı satırlarda iki profilin kararı 165 işlemde farklı (301 işlem validation'ın tamamında). Değişimin dağılımı:

- no_rule_match -> review: 138 işlem
- monitor -> no_rule_match: 27 işlem

## Geliştirme verisinde iki politikanın karar metrikleri

| Politika | İnceleme | TP | FP | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| raw (varsayılan) + kurallar | 3,805 | 251 | 3,554 | %6.597 | %12.179 |
| behavior_context + kurallar | 3,667 | 236 | 3,431 | %6.436 | %11.451 |
| raw skor >= 0.85 | 2,273 | 177 | 2,096 | %7.787 | %8.588 |
| context skor >= 0.85 | 2,135 | 162 | 1,973 | %7.588 | %7.860 |

Varsayılanı `raw` yapmak bu segmentte inceleme sayısını +138 değiştiriyor ve yakalanan fraud'u +15. Bunu iyileşme diye sunmuyorum: context indirimi kaldırıldığı için daha çok işlem 0.85 eşiğini geçiyor, yani iki politika eşit sayıda işlem incelemeye göndermiyor. Aradaki farkın bir kısmı sıralama değil, hacim.

Hangi knob'a bağlı olduğunu göstermek için: ham skorlarda 2,135 incelemeye denk gelen eşik 0.854156. Bu sayı yalnızca alarm hacmi üzerinden bulundu, etiket kullanılmadı ve R09 değiştirilmedi.

Bu eşik gerçekte 2,135 alarm üretiyor, yani bu segmentte hedefi tam tutuyor. Bunu genel bir garanti olarak yazmıyorum. Tek bir eşik, eşit skorlu satırlar olduğunda tam kapasite garanti etmez; k'ncı skora eşit başka satırlar varsa hepsi eşiği geçer. Tam olarak k işlem seçmek gerekiyorsa yol eşik değil, projenin her yerinde kullandığı TransactionID kırılımlı top-k seçimidir. Eşiği bu yüzden bir öneri olarak değil, kararın hangi knob'a bağlı olduğunu göstermek için yazıyorum.

Kapasite politikası verilmediği için R09'u yeniden kalibre etmedim; bu bir seçim adımı olur ve neye göre seçtiğimi yazamazdım.

## Politikaların durumu

| Politika | Nasıl ölçüldü | Serviste yeri |
|---|---|---|
| Ham skor + kurallar | Bu raporda, geliştirme verisinde | Varsayılan (`raw`) |
| Davranış context'i + kurallar | Final raporda ve kural raporunda | İsteğe bağlı (`behavior_context`) |
| Ürün riski + kurallar | Geliştirme adayı, belirsizliği ölçüldü | Önerilmiyor (`product_risk`) |

Final rapordaki kural satırları ikinci satıra aittir. Final testi yeniden açmadım, bu yüzden varsayılan politikanın final dönemindeki kural performansını bilmiyorum ve bilmediğimi yazıyorum. Ölçebildiğim şey geliştirme verisindeki fark.

Yeniden çalıştırma: `python scripts/evaluate_default_policy.py`. Çıktı: `artifacts/default_policy/evaluation.json`.
