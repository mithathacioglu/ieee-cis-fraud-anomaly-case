# Rule engine

JSON kuralları feature ve anomali skorlarından inceleme kararı üretir. Raw/context skorları değişmez. Kurallar fraud etiketi üretmez; `review` inceleme önerisi, `monitor` izleme önerisidir. `no_rule_match` güvenli/izin verilmiş işlem anlamına gelmez.

## Kural sözleşmesi

`config/rules.json`: version, default_action ve rules. Her kural id, enabled, priority, description, when ve then içerir. `all` / `any` iç içe kullanılabilir. Operatörler: eq, ne, gt, gte, lt, lte, in, is_missing, is_present. Yalnızca kayıtlı sayısal/boolean feature ve skor adları kullanılabilir; etiket, ID, Python ifadesi ve bilinmeyen alanlar reddedilir.

Yüksek sayısal priority kazanır. Eşitlikte alfabetik kural ID'si kazanır; JSON sırası sonucu değiştirmez. Bütün eşleşmeler ve kaybeden kurallar açıklamada tutulur. Review kuralları mevcut dosyada monitor kuralından yüksek önceliklidir; yeni bir yapılandırmada önceliklerin sorumluluğu politika sahibindedir.

Eksik hücre normal karşılaştırmalarda false olur; NaN != eşik de eşleşmez. Eksikliği aramak için is_missing açıkça seçilir. Eksik kolon, hatalı yapılandırma sayılır ve hata verir. Devre dışı kuralın girdisi aranmaz. Koşullar arasında kısa devreyle açıklama atlanmaz; bütün koşulların gözlenen değeri raporlanır.

Motor her başlatmada JSON'u doğrular ve değişmez bir kopyasını alır. Dosyayı değiştirdikten sonra yeni RuleEngine.from_json(...) örneği yeni politikayı çalıştırır; mevcut istek ortasında politika değişmez. Otomatik dosya izleyici yoktur. Yapılandırma özeti her açıklamada bulunur.

## Kurallar ve çalışma kapsamı

Eşikler örnek iş politikası hipotezleridir; kurum tarafından verilmiş veya fraud başarısına göre optimize edilmiş eşikler değildir. Ülke veya gerçek yerel gece bilgisi olmadığı için bu tür kurallar uydurulmadı. Kart/adres entity'si gerçek kişi, DeviceInfo da benzersiz cihaz kimliği değildir. Rare-pair kurallarındaki 1000 geçmiş işlem şartı global başlangıç filtresidir; her kategori çifti için 1000 gözlem anlamına gelmez.

| Kural | Öncelik | Karar | Tüm veride eşleşme | Validation eşleşme | Validation kazanan |
|---|---:|---|---:|---:|---:|
| R01_multi_entity_deviation | 100 | review | 15 | 4 | 4 |
| R02_hourly_velocity | 95 | review | 1,294 | 4 | 4 |
| R03_daily_velocity | 90 | review | 4,106 | 241 | 240 |
| R04_amount_jump | 85 | review | 13,206 | 1,283 | 1,271 |
| R05_new_device_anomaly | 80 | review | 305 | 32 | 32 |
| R06_new_product_and_email | 75 | review | 1,171 | 50 | 47 |
| R07_rare_product_card_amount | 70 | review | 153 | 17 | 14 |
| R08_rare_email_pair_anomaly | 65 | review | 1,712 | 115 | 110 |
| R09_high_adjusted_score | 60 | review | 26,922 | 2,135 | 1,945 |
| R10_familiar_activity | 10 | monitor | 91,243 | 11,476 | 11,412 |

Koşulların tam değerleri JSON'dadır. Bir işlem birden fazla kuralla eşleşebilir; eşleşme sayıları toplanarak işlem sayısı bulunamaz.

## Geliştirme verisinde karar etkisi

Context geliştirmesinde görülmüş sonraki validation bölümü kullanıldı; bağımsız final holdout değildir. Kural eşikleri bu çalıştırmanın etiketli sonuçlarına bakılarak değiştirilmedi. Final test etiketleri okunmadı. Raw/context karşılaştırma eşiği sabit 0.85'tir; önceki context raporunun train quantile eşiğiyle birebir aynı değildir.

Buradaki `rule_review` satırı `data/processed/context_scores.parquet` üzerinden üretildi, yani davranış context'i uygulanmış skorlara ait. R09 eşiği `adjusted_anomaly_score` alanında çalıştığı için servisin varsayılan `raw` profili aynı kararları vermiyor; aynı JSON kurallarını çalıştırmak aynı kararı üretmek anlamına gelmez. İki profilin farkı [varsayılan politika raporunda](default_policy.md) ölçüldü.

| Politika | İnceleme | TP | FP | FN | Precision | Recall | FPR |
|---|---:|---:|---:|---:|---:|---:|---:|
| raw_at_0_85 | 2,273 | 177 | 2,096 | 1,884 | %7.787 | %8.588 | %3.678 |
| context_at_0_85 | 2,135 | 162 | 1,973 | 1,899 | %7.588 | %7.860 | %3.462 |
| rule_review | 3,667 | 236 | 3,431 | 1,825 | %6.436 | %11.451 | %6.020 |

Bu politikalar eşit sayıda işlem incelemeye göndermez. Daha çok fraud yakalama, daha yüksek inceleme yükünden kaynaklanabilir; tek başına model/sıralama iyileşmesi kanıtı değildir. Rule engine bir sıralama skoru tanımlamaz; bu nedenle karar etiketlerinden yapay AP/ROC üretilmez. Operasyon kapasitesi ve fraud maliyeti verilmeden üretimde hangi politikanın kullanılacağına karar verilemez.

Validation bölümünde 265 işlem birden fazla kuralla eşleşti; 64 işlemde review/monitor çakışması çözüldü.

## Gerçek işlem açıklamaları

- action_conflict: TransactionID 3400481; eşleşmeler R03_daily_velocity, R10_familiar_activity; kazanan R03_daily_velocity; karar review.
- multiple_matches: TransactionID 3400401; eşleşmeler R08_rare_email_pair_anomaly, R09_high_adjusted_score; kazanan R08_rare_email_pair_anomaly; karar review.
- review: TransactionID 3400401; eşleşmeler R08_rare_email_pair_anomaly, R09_high_adjusted_score; kazanan R08_rare_email_pair_anomaly; karar review.
- monitor: TransactionID 3400382; eşleşmeler R10_familiar_activity; kazanan R10_familiar_activity; karar monitor.
- no_rule_match: TransactionID 3400378; eşleşmeler yok; kazanan yok; karar no_rule_match.

Her açıklamada koşul ağacı, eşik, gözlenen değer, eksiklik, selected/superseded/not_matched/disabled durumu ve yapılandırma SHA-256 bulunur. Ayrıntılar `artifacts/rules/examples.json` içindedir.

## Yeniden üretim

```powershell
.\.venv\Scripts\python.exe -m fraud_case.evaluate_rules
```

Özel politika: `--rules config/my_rules.json`. Yeniden çalıştırma yerel rules çıktılarının üstüne yazar. Son kullanılan politikanın kopyası `artifacts/rules/rules_snapshot.json` içinde saklanır.

- `data/processed/rule_decisions.parquet`: ID/split, kural eşleşmeleri, kazanan, karar ve çakışma bayrağı.
- `artifacts/rules/evaluation.json`: politika özeti, kaynak dosya hash'leri, sayımlar ve karar metrikleri.
- `tests/test_rules.py`: öncelik, çakışma, eşitlik, eksik veri, doğrulama, bütün kuralların erişilebilirliği ve açıklama testleri.
