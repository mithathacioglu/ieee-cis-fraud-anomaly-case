# Final test değerlendirmesi

Son kronolojik bölümde 118,108 işlem. Model, context ve rule yapılandırması ile kaynak kod hash'leri test etiketleri okunmadan önce kaydedildi. Bu sonuçlara göre parametre ayarı yapılmadı.

Train'den sabitlenen eşik: 0.85167813. Validation'da seçilen context gücü değişmedi. Geçmiş özellikleri kronolojik ve etiketsiz güncellenir; model ve referanslar train'de sabittir.

| Politika | İnceleme | TP | FP | Precision | Recall | FPR | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| Raw | 4,681 | 487 | 4,194 | %10.404 | %11.983 | %3.678 | %6.717 |
| Context | 4,517 | 459 | 4,058 | %10.162 | %11.294 | %3.558 | %6.673 |
| Rules | 8,966 | 581 | 8,385 | %6.480 | %14.296 | %7.352 | n/a |

Aynı 4,681 inceleme kapasitesinde raw 487, context 475 fraud yakaladı.

Rules kararları skor sıralaması değildir ve kendi 0.85 yüksek-skor kuralı dahil farklı inceleme bütçesi kullanır; tablodan doğrudan ranking üstünlüğü çıkarılmaz. AP yalnızca sürekli raw/context skorlarına uygulanmıştır. Gerçek fraud oranı ve TP/FP sayıları, küçük mutlak recall farklarıyla birlikte değerlendirilmelidir.

## Bu tablo hangi politikaya ait

Üç satır da bu test koşulduğu andaki politikayı ölçüyor: davranış context'i seçilmiş 0.5 gücüyle açık. **Context** satırı doğrudan o skoru, **Rules** satırı da o skorlardan üretilen kural kararlarını gösteriyor; R09 eşiği `adjusted_anomaly_score` üzerinde çalıştığı için Rules satırı context'ten bağımsız değil.

Servisin şu andaki varsayılanı `raw`, yani context indirimi uygulanmıyor. Dolayısıyla **Rules** satırı güncel varsayılanın final dönemindeki performansı değildir. Bunu ölçmek final testi yeniden açmak olurdu; açmadım. Ölçebildiğim şey aynı değişimin geliştirme verisindeki etkisiydi: iki profil sonraki validation yarısında 165 işlemde farklı karar veriyor ([varsayılan politika raporu](default_policy.md)). Final dönemi için bu farkı bilmiyorum ve tahmin etmiyorum.

**Raw** satırı context'ten etkilenmez; güncel varsayılanın skor tarafı bu satırdır.

## Yorum

Bu çalışma gözetimsiz anomali sıralaması ve açıklanabilir iş kurallarını birleştirir. Fraud etiketiyle optimize edilmiş bir sınıflandırıcı veya üretime hazır fraud performansı iddiası taşımaz. Daha fazla inceleme ile daha fazla fraud bulmak tek başına verimlilik artışı değildir. Bir sonraki model revizyonu yeni bir değerlendirme planı gerektirir; bu test artık görülmüştür.

Kaynak snapshot: artifacts/final/frozen_manifest.json. Tüm metrikler: artifacts/final/evaluation.json. Bu metin `python scripts/evaluate_final.py --report-only` ile aynı JSON'dan yeniden üretilir; sonuç dosyası değişmez.
