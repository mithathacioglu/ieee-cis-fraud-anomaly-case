# Ürün riski kazanımının belirsizliği

Ürün riski adayı ham skora göre %5 (+12), %10 (+94) bütçesinde daha fazla; %1 (-9) bütçesinde daha az fraud yakalıyor. Farkların yönü kapasiteye göre değiştiği için "kazandı" demeden önce her bütçede bu farkın ne kadar oynadığını ölçtüm. Kullandığım yöntem kronolojik deneydeki eşlenmiş blok bootstrap'ının aynısı; iki deney de `fraud_case.evaluation.paired_block_intervals` fonksiyonunu çağırıyor.

Segment: validation'ın audit yarısı, 59,054 işlem ve 2,061 fraud. Bu bölüm context geliştirmesi sırasında zaten görüldü; bağımsız holdout değil.

## Aynı bütçede gözlenen fark

| Bütçe | İnceleme | Ham | Davranış context | Ürün context | Ürün - Ham |
|---|---:|---:|---:|---:|---:|
| %1 | 591 | 46 | 46 | 37 | -9 |
| %5 | 2,953 | 230 | 227 | 242 | +12 |
| %10 | 5,906 | 457 | 458 | 551 | +94 |

## Eşlenmiş blok bootstrap

Bloklar saat sınırına göre kesiliyor, gözlenen blok sayısı kadar blok yerine koymayla çekiliyor ve her çekimde bütçe satır sayısına göre yeniden hesaplanıyor. Aynı çekim bütün sıralamalara uygulandığı için farklar eşlenmiş kalıyor. Aralıklar yüzdelik yöntemiyle %95.

| Blok | Bütçe | Karşılaştırma | Gözlenen | %95 aralık | Sıfırı dışlıyor |
|---|---|---|---:|---|---|
| 6 saat | %1 | behavior_minus_raw | +0 | [-1.8, 4.1] | hayır |
| 6 saat | %1 | product_minus_raw | -9 | [-34.4, 12.8] | hayır |
| 6 saat | %1 | product_minus_behavior | -9 | [-34.4, 12.5] | hayır |
| 6 saat | %5 | behavior_minus_raw | -3 | [-9.0, 9.7] | hayır |
| 6 saat | %5 | product_minus_raw | +12 | [-10.6, 44.7] | hayır |
| 6 saat | %5 | product_minus_behavior | +15 | [-13.0, 46.0] | hayır |
| 6 saat | %10 | behavior_minus_raw | +1 | [0.0, 2.2] | hayır |
| 6 saat | %10 | product_minus_raw | +94 | [29.9, 159.6] | evet |
| 6 saat | %10 | product_minus_behavior | +93 | [28.7, 159.6] | evet |
| 24 saat | %1 | behavior_minus_raw | +0 | [-3.6, 3.8] | hayır |
| 24 saat | %1 | product_minus_raw | -9 | [-37.1, 14.6] | hayır |
| 24 saat | %1 | product_minus_behavior | -9 | [-37.2, 14.1] | hayır |
| 24 saat | %5 | behavior_minus_raw | -3 | [-8.5, 9.2] | hayır |
| 24 saat | %5 | product_minus_raw | +12 | [-6.9, 47.1] | hayır |
| 24 saat | %5 | product_minus_behavior | +15 | [-9.6, 49.5] | hayır |
| 24 saat | %10 | behavior_minus_raw | +1 | [0.0, 2.9] | hayır |
| 24 saat | %10 | product_minus_raw | +94 | [27.9, 169.0] | evet |
| 24 saat | %10 | product_minus_behavior | +93 | [27.0, 169.0] | evet |

## Dönem tutarlılığı

Aynı aritmetiği audit yarısının iki kronolojik parçasında ayrı ayrı hesapladım. Tek bir pencerede sıfırdan ayrılan bir fark, dönem değiştiğinde yön değiştiriyorsa karar için yeterli değildir.

| Bütçe | Karşılaştırma | İlk yarı | İkinci yarı | Aynı yön |
|---|---|---:|---:|---|
| %1 | behavior_minus_raw | +1 | +0 | biri sıfır |
| %1 | product_minus_raw | +2 | -13 | ters |
| %1 | product_minus_behavior | +1 | -13 | ters |
| %5 | behavior_minus_raw | +0 | +2 | biri sıfır |
| %5 | product_minus_raw | +4 | +19 | aynı |
| %5 | product_minus_behavior | +4 | +17 | aynı |
| %10 | behavior_minus_raw | +0 | +0 | biri sıfır |
| %10 | product_minus_raw | +55 | +33 | aynı |
| %10 | product_minus_behavior | +55 | +33 | aynı |

## Karar

Cevap bütçeden bağımsız değil, o yüzden tek cümlede vermiyorum. Ürün riski ile ham skor arasındaki fark için üç kapasitede durum şu:

| Bütçe | Gözlenen | Hesaplanan blok | Aralıklar sıfırı dışlıyor mu | İki yarıda yön | Okuma |
|---|---:|---:|---|---|---|
| %1 | -9 | 2/2 | hayır, sıfırı kapsıyor | ters | kanıt yok |
| %5 | +12 | 2/2 | hayır, sıfırı kapsıyor | aynı | yön tutarlı ama belirsizlik sıfırı kapsıyor |
| %10 | +94 | 2/2 | evet, pozitif yönde | aynı | bu pencerede kazanım var |

"Model iyi mi kötü mü" sorusunun bu veriyle tek cevabı yok; cevap kapasiteye göre değişiyor. %10 kapasitede fark sıfırdan ayrılıyor ve iki dönemde de kazanım yönünde. %5 kapasitede yön tutarlı ama belirsizlik sıfırı kapsıyor; bu farka dayanarak politika değiştirmem. %1 kapasitede kanıt yok.

Bu nedenle ürün riskini varsayılan yapmıyorum ve "kazandı" da demiyorum. Önerim: kapasite politikası sabitlenmeden bu karar verilmemeli; hangi kapasitede çalışıldığı sabitlendikten sonra sonraki adım bu katmanı görülmemiş yeni bir dönemde ölçmek.

Bu ölçüm modeller ve politika sabitken, bu pencerede geçerli. Yeniden eğitim belirsizliğini kapsamıyor; bloklar arasında aynı kart/adres entity'sinin tekrar görünmesi bağımsızlık varsayımını zorluyor; segment zaten görülmüş geliştirme verisi. Bu üç sınır nedeniyle buradan üretim kararı çıkarmıyorum.

Yeniden çalıştırma: `python scripts/evaluate_product_uncertainty.py`. Çıktı: `artifacts/product_uncertainty/evaluation.json`.
