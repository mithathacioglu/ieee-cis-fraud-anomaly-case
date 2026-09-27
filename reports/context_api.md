# API bağlam senaryoları

Bu örnekler sentetiktir. Gerçek IEEE-CIS takvimi veya güven kaydı değildir; fraud başarısı ölçmez.
Kaydedilmiş gerçek anomali modeli ve seçilmiş context ayarları değiştirilmeden API üzerinden çalıştırılmıştır.

| Senaryo | Raw | Adjusted | Uygulanan bağlam |
|---|---:|---:|---|
| unknown_calendar | 0.402748 | 0.402748 | yok |
| weekday_business_hours | 0.402748 | 0.398103 | business_hours |
| weekend_without_schedule | 0.402748 | 0.402748 | yok |
| weekend_with_schedule | 0.402748 | 0.398103 | expected_weekend |
| prior_trust | 0.402748 | 0.391036 | verified_trust |
| same_second_trust | 0.402748 | 0.402748 | yok |

Altı senaryoda raw skor aynıdır. Yalnızca geçerli bağlam ilgili katkıyı azaltır; aynı saniyede alınan güven kaydı geçmiş bilgi sayılmaz.
64 kayıtlı işlemde API'nin 15 skor alanı, değişiklik öncesi kaydedilen sonuçlarla tolerans içinde eşleşmiştir.

Bu tekrarın kapsamını olduğundan geniş göstermemek gerekir: kayıtlı feature'lardan ve kaydedilmiş modelden başlıyor, yani skorlama, birleştirme ve context katmanını doğruluyor. Feature üretimini yeniden türetmiyor; dolayısıyla geçmiş feature'larının nasıl hesaplandığındaki bir değişikliği tek başına yakalamaz. O katman `scripts/verify_features.py` ile ayrı ve farklı bir yöntemle kontrol ediliyor, ayrıca dondurulmuş parquet hash'leri sonucun değişmediğini gösteriyor.

## Varsayılan profil

Yukarıdaki tablo `behavior_context` profiliyle üretildi; `data/processed/context_scores.parquet` de o profile ait. Servisin varsayılanı `raw`: context indirimi uygulanmaz. Bunu iddia olarak bırakmıyorum, aynı koşuda ölçüyorum. Varsayılan profille altı senaryonun tamamında eşleşen kural listesi aynı kalır, uygulanan indirim boş ve adjusted skor raw skora eşittir; 64 kayıtlı işlemde de `context_reduction` sıfırdır.

| Senaryo | Eşleşen kural | İndirim |
|---|---|---:|
| unknown_calendar | yok | 0.000000 |
| weekday_business_hours | business_hours | 0.000000 |
| weekend_without_schedule | yok | 0.000000 |
| weekend_with_schedule | expected_weekend | 0.000000 |
| prior_trust | verified_trust | 0.000000 |
| same_second_trust | yok | 0.000000 |

Kural eşleşmesini kapatmadım, yalnızca skora etkisini kapattım: inceleme ekibi hangi bağlamın tuttuğunu yine görür, karar skoru bundan etkilenmez.

Yeniden çalıştırma: `python scripts/verify_context_api.py`. Bu doğrulama fraud etiketi okumaz.
