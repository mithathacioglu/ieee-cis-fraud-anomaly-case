# Ürün riski profilinin servis doğrulaması

118.108 validation işleminin yeni context skorları bağımsız deney betiğiyle karşılaştırıldı. Davranış katman katkılarının toplamı + ürün riski düzeltmesi = final adjusted skor özdeşliği bütün satırlarda doğrulandı. 64 işlemde gerçek model kullanan API skoru, açıklaması ve kural kararı batch çıktısıyla eşleşti. Bu kontrol yeni final test değildir.

## Aynı 10 kuralın sonraki validation etkisi

| Profil | İnceleme | TP | FP | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| baseline | 3667 | 236 | 3431 | 6.436% | 11.451% |
| product_risk | 5275 | 382 | 4893 | 7.242% | 18.535% |

Kuralların sayısı ve eşikleri değiştirilmedi; R09 hâlâ adjusted>=0.85 kullanır. Yeni context skor dağılımı bu kuralın alarm sayısını değiştirebilir. Bu tablo farklı inceleme bütçeleri içerdiğinden, doğrudan sıralama üstünlüğü anlamına gelmez. Skorun sabit bütçeli karşılaştırması [ürün context raporundadır](product_context.md).

Yeniden çalıştırma: `python scripts/verify_product_profile.py`. Gerçek HTTP ve LLM kontrolü ayrıca `python scripts/verify_http.py --context-profile product_risk` ile yapılır.
