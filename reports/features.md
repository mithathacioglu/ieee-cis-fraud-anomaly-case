# Feature engineering

590,540 işlem için 41 özellik üretildi.

Hedef (`isFraud`) dosyadan dahi okunmaz. `TransactionID` ve `split` yalnızca eşleme metadatasıdır; feature sözleşmesinde yoktur.

Her zaman grubunda önce bütün feature'lar hesaplanır, ardından geçmiş güncellenir. Validation ve test sırasında da yalnızca daha önce gözlenen etiketsiz işlemler geçmişi günceller. Model parametrelerinin bu bölümlerde yeniden fit edilmesi anlamına gelmez.

## Takvim bağlamı

Açık başlangıç referansı: `None`.

Referans yoksa gerçek hour/day_of_week/is_weekend/is_business_hours alanları null kalır. Relative day, 24 saatlik faz ve sin/cos alanları yine kullanılabilir. Açık referans verilirse takvim hesapları bu varsayıma dayanır; IEEE-CIS tarafından doğrulanmış tarih olarak sunulamaz.

## Kapsama

| Bölüm | İşlem | Tam entity vekili | En az 5 geçmiş işlem |
|---|---:|---:|---:|
| train | 354,324 | %87.04 | %63.67 |
| validation | 118,108 | %86.42 | %75.96 |
| test | 118,108 | %88.20 | %79.38 |

## Özellik sözleşmesi

| Özellik | Grup | Tanım |
|---|---|---|
| `amount` | column | Current transaction amount; original units. |
| `log_amount` | column | log(1 + current transaction amount). |
| `relative_day` | temporal | Elapsed seconds // 86400; not a calendar day. |
| `hour_phase` | temporal | Hour within the undisclosed reference's 24-hour cycle. |
| `hour_sin` | temporal | Sine of the relative 24-hour phase. |
| `hour_cos` | temporal | Cosine of the relative 24-hour phase. |
| `hour` | temporal | Calendar hour; null unless an explicit reference is supplied. |
| `day_of_week` | temporal | Monday=0; null without an explicit reference. |
| `is_weekend` | context | Saturday/Sunday; null without an explicit reference. |
| `is_business_hours` | context | Weekday 09:00-17:00 in the configured fixed offset; otherwise null. |
| `calendar_available` | context | An explicit timezone-aware reference was supplied. |
| `has_identity` | context | Identity row exists; not evidence of trust. |
| `entity_known` | entity | All five card/address proxy components are present. |
| `prior_transaction_count` | entity | Number of strictly earlier transactions for the proxy. |
| `prior_mean_amount` | entity | Mean amount of strictly earlier proxy transactions. |
| `prior_std_amount` | entity | Population standard deviation of earlier amounts; null before two observations. |
| `amount_to_prior_mean` | entity | Current amount / prior mean; null when mean is zero or absent. |
| `amount_zscore` | entity | Signed deviation / prior std; requires five observations and positive std. |
| `prior_count_1h` | temporal | Proxy transactions in [t-3600, t). |
| `prior_count_24h` | temporal | Proxy transactions in [t-86400, t). |
| `seconds_since_previous` | temporal | Time since the most recent strictly earlier proxy transaction. |
| `history_span_days` | entity | Elapsed time since the proxy's first earlier observation. |
| `prior_distinct_products` | relational | Distinct observed products in proxy history. |
| `prior_distinct_email_domains` | relational | Distinct payer email domains in proxy history. |
| `prior_distinct_devices` | relational | Distinct observed device descriptions in proxy history. |
| `entity_product_is_new` | relational | Current product absent from nonempty observed product history; otherwise null. |
| `entity_email_is_new` | relational | Current payer domain absent from nonempty observed domain history; otherwise null. |
| `entity_device_is_new` | relational | Current device description absent from nonempty observed device history; otherwise null. |
| `entity_hour_probability` | temporal | Laplace-smoothed prior hour-phase frequency; null for no history. |
| `prior_global_count` | relational | Number of strictly earlier transactions across the stream. |
| `product_prior_frequency` | relational | Earlier product count / earlier rows with observed product. |
| `email_prior_frequency` | relational | Earlier payer domain count / earlier rows with observed payer domain. |
| `device_prior_frequency` | relational | Earlier device count / earlier rows with observed device description. |
| `product_card_prior_frequency` | relational | Earlier product/card-brand pair count / complete earlier pairs. |
| `email_pair_prior_frequency` | relational | Earlier payer/recipient domain pair count / complete earlier pairs. |
| `product_card_prior_support` | relational | Number of earlier observations supporting the current product/card pair. |
| `email_pair_prior_support` | relational | Number of earlier observations supporting the current email pair. |
| `email_domains_match` | relational | Payer and recipient email domains match; null if either is absent. |
| `history_sufficient` | context | Complete proxy with at least five earlier transactions. |
| `frequent_entity` | context | At least 20 earlier transactions and at least seven days of observed history; not a trust label. |
| `usual_amount` | context | At least five earlier observations and amount/prior mean between 2/3 and 1.5; otherwise null or false. |

## Sınırlar

Entity, önceki adımlardaki kart/adres vekilidir. DeviceInfo benzersiz bir cihaz ID'si değil, paylaşılan cihaz açıklaması olabilir. E-posta alanı değişimi kişinin e-posta adresinin değiştiğini kanıtlamaz.

Geçmişsiz entity için count=0 ve ortalama=null; entity tanımsızsa count da null'dır. İlk gözlem yeni ilişki anomalisine dönüştürülmez. Sık işlem yapmak güvenilirlik etiketi değildir. Skorlama katmanında minimum geçmiş ve eksiklikler ayrıca ele alınmalıdır.

Velocity pencereleri sol sınırı dahil, şimdiki zamanı hariç tutar: [t-3600,t) ve [t-86400,t).

FeatureBuilder tek yazarlı bir stream nesnesidir. Yeni batch önceki batch'in son zamanından büyük bir zamanda başlamalıdır; aynı saniye iki batch'e bölünemez. Geç gelen olaylar reddedilir; yeniden sıralama/geri sarma bu prototipte yoktur.
