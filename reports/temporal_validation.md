# Kronolojik model karşılaştırması

Train ve validation üzerinde 3 değerlendirme dönemi kullanıldı. Her dönemde modeller yeniden eğitildi. Final test bu çalışmada kullanılmadı. Bu veri bölümleri daha önce incelendiği için sonuçlar geliştirme sonuçlarıdır.

Deney düzeni, varsayımlar ve tekrar çalıştırma komutu [protokolde](../docs/validation_protocol.md). Eşlenmiş modeller anomali motorunun kaynak feature'larını görür. Dönem başına kolon listeleri JSON çıktısında kayıtlıdır. Bu koşuda girdi sayıları: full = [36]; without_time = [34]; matched = [26].

## %5 inceleme bütçesi

Etiket gecikmesi, eğitimde hangi işlemlerin kullanılabildiğini belirleyen bir varsayımdır. 0, 7 gün koşuları aynı değerlendirme işlemlerini içerir; sonuçları birbirine eklenmez.

| Gecikme (gün) | Dönem | İşlem | Fraud | İnceleme | Ham | GB eşlenmiş | GB tam | GB iki kolon çıkarılmış | Hibrit |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 39,259 | 1,588 | 1,963 | 213 | 459 | 457 | 474 | 349 |
| 0 | 2 | 42,992 | 1,744 | 2,150 | 194 | 509 | 539 | 615 | 389 |
| 0 | 3 | 35,857 | 1,279 | 1,793 | 139 | 390 | 423 | 422 | 305 |
| 7 | 1 | 39,259 | 1,588 | 1,963 | 210 | 407 | 455 | 410 | 371 |
| 7 | 2 | 42,992 | 1,744 | 2,150 | 198 | 537 | 554 | 516 | 445 |
| 7 | 3 | 35,857 | 1,279 | 1,793 | 132 | 381 | 435 | 399 | 301 |

0 gün gecikmede toplam 5,906 incelemede ham skor 546, eşlenmiş GB 1358, tam GB 1419 fraud yakaladı. Eşlenmiş GB ile ham skor arasındaki dönemlik farklar: +246, +315, +251. Hibrit, eşlenmiş GB'ye göre toplam -315 TP farkı verdi.

7 gün gecikmede toplam 5,906 incelemede ham skor 540, eşlenmiş GB 1325, tam GB 1444 fraud yakaladı. Eşlenmiş GB ile ham skor arasındaki dönemlik farklar: +197, +339, +249. Hibrit, eşlenmiş GB'ye göre toplam -208 TP farkı verdi.

Toplamlar dönem başına ayrı uygulanan bütçelerin toplamıdır. Model ve girdi seçiminin etkileri birlikte görülür; eşlenmiş koşul dahi farkın yalnız etiketlerden kaynaklandığını kanıtlamaz. İşlem başına maliyet bilgisi olmadığı için TP farkı parasal getiriye çevrilmedi.

## Dönem ve eğitim kapsamı

Zamanlar veri setinin açıklanmayan başlangıcından itibaren saniyedir; takvim tarihi değildir.

| Gecikme | Dönem | Eğitim işlemi | Eğitim bitişi | Değerlendirme başlangıcı | Değerlendirme bitişi | Fraud oranı |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 354,324 | 8745772 | 8745798 | 9894811 | %4.04 |
| 0 | 2 | 393,583 | 9894811 | 9894826 | 11043792 | %4.06 |
| 0 | 3 | 436,575 | 11043792 | 11043833 | 12192842 | %3.57 |
| 7 | 1 | 331,489 | 8140967 | 8745798 | 9894811 | %4.04 |
| 7 | 2 | 372,738 | 9289925 | 9894826 | 11043792 | %4.06 |
| 7 | 3 | 413,470 | 10439022 | 11043833 | 12192842 | %3.57 |

## Bütün bütçeler ve sıralama metrikleri

Hibrit satırı, sabit kota ile oluşturulan inceleme sırasını ölçer; bir olasılık modelinin skoru değildir.

| Gecikme | Dönem | Model / girdi | AP | ROC AUC | TP %1 | TP %5 | TP %10 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 0 | 1 | Ham anomali | 0.0832 | 0.7040 | 33 | 213 | 396 |
| 0 | 1 | LR / tam | 0.1466 | 0.7612 | 108 | 386 | 621 |
| 0 | 1 | GB / tam | 0.2131 | 0.8143 | 159 | 457 | 733 |
| 0 | 1 | LR / iki kolon çıkarılmış | 0.1554 | 0.7689 | 123 | 392 | 632 |
| 0 | 1 | GB / iki kolon çıkarılmış | 0.2100 | 0.8079 | 164 | 474 | 723 |
| 0 | 1 | LR / eşlenmiş | 0.1456 | 0.7728 | 103 | 386 | 626 |
| 0 | 1 | GB / eşlenmiş | 0.1949 | 0.8074 | 146 | 459 | 698 |
| 0 | 1 | Hibrit / %50 kota | 0.1478 | 0.7911 | 101 | 349 | 629 |
| 0 | 2 | Ham anomali | 0.0698 | 0.6713 | 32 | 194 | 341 |
| 0 | 2 | LR / tam | 0.1715 | 0.7832 | 130 | 523 | 787 |
| 0 | 2 | GB / tam | 0.2024 | 0.8247 | 142 | 539 | 880 |
| 0 | 2 | LR / iki kolon çıkarılmış | 0.1691 | 0.7837 | 131 | 497 | 779 |
| 0 | 2 | GB / iki kolon çıkarılmış | 0.2095 | 0.8213 | 149 | 615 | 864 |
| 0 | 2 | LR / eşlenmiş | 0.1707 | 0.7897 | 143 | 510 | 771 |
| 0 | 2 | GB / eşlenmiş | 0.1856 | 0.8128 | 131 | 509 | 860 |
| 0 | 2 | Hibrit / %50 kota | 0.1333 | 0.7864 | 77 | 389 | 680 |
| 0 | 3 | Ham anomali | 0.0649 | 0.6721 | 32 | 139 | 296 |
| 0 | 3 | LR / tam | 0.1387 | 0.7524 | 115 | 335 | 478 |
| 0 | 3 | GB / tam | 0.1846 | 0.8040 | 127 | 423 | 603 |
| 0 | 3 | LR / iki kolon çıkarılmış | 0.1354 | 0.7518 | 108 | 338 | 480 |
| 0 | 3 | GB / iki kolon çıkarılmış | 0.1767 | 0.8047 | 113 | 422 | 587 |
| 0 | 3 | LR / eşlenmiş | 0.1348 | 0.7539 | 110 | 335 | 470 |
| 0 | 3 | GB / eşlenmiş | 0.1727 | 0.8029 | 123 | 390 | 592 |
| 0 | 3 | Hibrit / %50 kota | 0.1262 | 0.7777 | 83 | 305 | 508 |
| 7 | 1 | Ham anomali | 0.0830 | 0.7021 | 38 | 210 | 400 |
| 7 | 1 | LR / tam | 0.1435 | 0.7524 | 109 | 383 | 626 |
| 7 | 1 | GB / tam | 0.1966 | 0.7952 | 151 | 455 | 709 |
| 7 | 1 | LR / iki kolon çıkarılmış | 0.1511 | 0.7613 | 116 | 393 | 634 |
| 7 | 1 | GB / iki kolon çıkarılmış | 0.1870 | 0.7963 | 146 | 410 | 659 |
| 7 | 1 | LR / eşlenmiş | 0.1444 | 0.7698 | 102 | 383 | 616 |
| 7 | 1 | GB / eşlenmiş | 0.1864 | 0.7879 | 150 | 407 | 670 |
| 7 | 1 | Hibrit / %50 kota | 0.1462 | 0.7822 | 109 | 371 | 567 |
| 7 | 2 | Ham anomali | 0.0710 | 0.6701 | 35 | 198 | 353 |
| 7 | 2 | LR / tam | 0.1701 | 0.7778 | 131 | 510 | 774 |
| 7 | 2 | GB / tam | 0.2015 | 0.8153 | 155 | 554 | 856 |
| 7 | 2 | LR / iki kolon çıkarılmış | 0.1684 | 0.7809 | 127 | 495 | 772 |
| 7 | 2 | GB / iki kolon çıkarılmış | 0.1911 | 0.8119 | 161 | 516 | 810 |
| 7 | 2 | LR / eşlenmiş | 0.1714 | 0.7882 | 143 | 510 | 766 |
| 7 | 2 | GB / eşlenmiş | 0.1978 | 0.7996 | 156 | 537 | 826 |
| 7 | 2 | Hibrit / %50 kota | 0.1430 | 0.7819 | 92 | 445 | 721 |
| 7 | 3 | Ham anomali | 0.0641 | 0.6708 | 34 | 132 | 288 |
| 7 | 3 | LR / tam | 0.1372 | 0.7489 | 113 | 335 | 478 |
| 7 | 3 | GB / tam | 0.1997 | 0.8040 | 145 | 435 | 609 |
| 7 | 3 | LR / iki kolon çıkarılmış | 0.1331 | 0.7491 | 102 | 328 | 481 |
| 7 | 3 | GB / iki kolon çıkarılmış | 0.1833 | 0.8013 | 135 | 399 | 586 |
| 7 | 3 | LR / eşlenmiş | 0.1345 | 0.7531 | 109 | 337 | 468 |
| 7 | 3 | GB / eşlenmiş | 0.1794 | 0.8034 | 139 | 381 | 583 |
| 7 | 3 | Hibrit / %50 kota | 0.1303 | 0.7788 | 79 | 301 | 518 |

## Zaman bloklarıyla bootstrap

Aralıklar %5 bütçedeki TP farkına aittir. Örneklerin işlem sayısı değiştiği için fark önce işlem başına hesaplanıp özgün pencere büyüklüğüne ölçeklenir. Her iki model aynı örnekle değerlendirilir. Modeller bootstrap içinde yeniden eğitilmez; bloklar arası entity bağımlılığı korunmaz. Bu aralıklar gelecekteki performansın garantisi veya çoklu karşılaştırma düzeltmesi yapılmış testler değildir.

| Gecikme | Dönem | Blok (saat) | Blok sayısı | Karşılaştırma | Gözlenen TP farkı | %95 yüzdelik aralık |
|---:|---:|---:|---:|---|---:|---:|
| 0 | 1 | 6 | 54 | GB eşlenmiş − ham | +246 | [170.3, 311.4] |
| 0 | 1 | 6 | 54 | GB tam − ham | +244 | [181.9, 321.8] |
| 0 | 1 | 6 | 54 | GB iki kolon çıkarılmış − tam | +17 | [-23.6, 50.5] |
| 0 | 1 | 6 | 54 | LR eşlenmiş − ham | +173 | [93.1, 240.9] |
| 0 | 1 | 24 | 14 | GB eşlenmiş − ham | +246 | [150.6, 325.2] |
| 0 | 1 | 24 | 14 | GB tam − ham | +244 | [164.1, 327.8] |
| 0 | 1 | 24 | 14 | GB iki kolon çıkarılmış − tam | +17 | [-20.6, 48.6] |
| 0 | 1 | 24 | 14 | LR eşlenmiş − ham | +173 | [73.8, 257.9] |
| 0 | 2 | 6 | 54 | GB eşlenmiş − ham | +315 | [207.4, 421.2] |
| 0 | 2 | 6 | 54 | GB tam − ham | +345 | [236.2, 458.8] |
| 0 | 2 | 6 | 54 | GB iki kolon çıkarılmış − tam | +76 | [21.3, 113.6] |
| 0 | 2 | 6 | 54 | LR eşlenmiş − ham | +316 | [182.2, 444.7] |
| 0 | 2 | 24 | 14 | GB eşlenmiş − ham | +315 | [199.4, 445.8] |
| 0 | 2 | 24 | 14 | GB tam − ham | +345 | [252.4, 460.0] |
| 0 | 2 | 24 | 14 | GB iki kolon çıkarılmış − tam | +76 | [17.4, 110.8] |
| 0 | 2 | 24 | 14 | LR eşlenmiş − ham | +316 | [169.0, 442.5] |
| 0 | 3 | 6 | 54 | GB eşlenmiş − ham | +251 | [182.3, 327.8] |
| 0 | 3 | 6 | 54 | GB tam − ham | +284 | [208.6, 370.8] |
| 0 | 3 | 6 | 54 | GB iki kolon çıkarılmış − tam | -1 | [-31.7, 30.6] |
| 0 | 3 | 6 | 54 | LR eşlenmiş − ham | +196 | [105.3, 299.3] |
| 0 | 3 | 24 | 14 | GB eşlenmiş − ham | +251 | [195.3, 301.7] |
| 0 | 3 | 24 | 14 | GB tam − ham | +284 | [221.6, 343.6] |
| 0 | 3 | 24 | 14 | GB iki kolon çıkarılmış − tam | -1 | [-29.9, 24.1] |
| 0 | 3 | 24 | 14 | LR eşlenmiş − ham | +196 | [119.7, 274.2] |
| 7 | 1 | 6 | 54 | GB eşlenmiş − ham | +197 | [126.2, 275.5] |
| 7 | 1 | 6 | 54 | GB tam − ham | +245 | [167.8, 323.3] |
| 7 | 1 | 6 | 54 | GB iki kolon çıkarılmış − tam | -45 | [-99.9, 1.0] |
| 7 | 1 | 6 | 54 | LR eşlenmiş − ham | +173 | [91.7, 240.5] |
| 7 | 1 | 24 | 14 | GB eşlenmiş − ham | +197 | [109.7, 286.9] |
| 7 | 1 | 24 | 14 | GB tam − ham | +245 | [139.0, 345.3] |
| 7 | 1 | 24 | 14 | GB iki kolon çıkarılmış − tam | -45 | [-97.3, 1.0] |
| 7 | 1 | 24 | 14 | LR eşlenmiş − ham | +173 | [75.0, 254.8] |
| 7 | 2 | 6 | 54 | GB eşlenmiş − ham | +339 | [216.5, 499.6] |
| 7 | 2 | 6 | 54 | GB tam − ham | +356 | [241.7, 459.0] |
| 7 | 2 | 6 | 54 | GB iki kolon çıkarılmış − tam | -38 | [-84.2, 6.3] |
| 7 | 2 | 6 | 54 | LR eşlenmiş − ham | +312 | [177.0, 437.7] |
| 7 | 2 | 24 | 14 | GB eşlenmiş − ham | +339 | [225.8, 514.6] |
| 7 | 2 | 24 | 14 | GB tam − ham | +356 | [262.0, 450.0] |
| 7 | 2 | 24 | 14 | GB iki kolon çıkarılmış − tam | -38 | [-66.1, 0.0] |
| 7 | 2 | 24 | 14 | LR eşlenmiş − ham | +312 | [158.5, 446.4] |
| 7 | 3 | 6 | 54 | GB eşlenmiş − ham | +249 | [181.3, 341.6] |
| 7 | 3 | 6 | 54 | GB tam − ham | +303 | [225.1, 391.4] |
| 7 | 3 | 6 | 54 | GB iki kolon çıkarılmış − tam | -36 | [-77.7, -1.9] |
| 7 | 3 | 6 | 54 | LR eşlenmiş − ham | +205 | [121.0, 302.6] |
| 7 | 3 | 24 | 14 | GB eşlenmiş − ham | +249 | [183.9, 314.3] |
| 7 | 3 | 24 | 14 | GB tam − ham | +303 | [230.3, 371.1] |
| 7 | 3 | 24 | 14 | GB iki kolon çıkarılmış − tam | -36 | [-85.0, 2.9] |
| 7 | 3 | 24 | 14 | LR eşlenmiş − ham | +205 | [131.9, 280.8] |

## Listelerin örtüşmesi

Her iki liste ayrı ayrı %5 bütçeye sahiptir. Birleşimin inceleme sayısı daha yüksektir; birleşimdeki ek yakalamalar aynı bütçedeki kazanım olarak okunmamalıdır. Aynı bütçedeki hibrit sonuçları ilk tabloda yer alır.

| Gecikme | Dönem | Ortak inceleme | Yalnız ham listedeki TP | Yalnız eşlenmiş GB listesindeki TP | Birleşim inceleme | Birleşim TP |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 1 | 326 | 126 | 372 | 3600 | 585 |
| 0 | 2 | 395 | 121 | 436 | 3905 | 630 |
| 0 | 3 | 306 | 82 | 333 | 3280 | 472 |
| 7 | 1 | 413 | 120 | 317 | 3513 | 527 |
| 7 | 2 | 308 | 135 | 474 | 3992 | 672 |
| 7 | 3 | 354 | 84 | 333 | 3232 | 465 |

## Kararın sınırı

Bu deney, tanımlı dönemler ve kapasite varsayımı altında model karşılaştırmasıdır. Etiket gecikmesi, inceleme maliyeti ve kaçırılan fraud kaybı gerçek operasyon verisiyle doğrulanmadan üretim kararı verilemez. İncelenmiş validation üzerinde yeni bir yöntem seçildiği için sonraki doğrulama yeni bir dönemde yapılmalıdır.

Sayısal özet: [temporal_validation_summary.json](temporal_validation_summary.json). Kaynak: `artifacts/temporal_validation/evaluation.json`. İşlem düzeyindeki tahminler aynı klasördeki `*_predictions.parquet` dosyalarında; çalıştırma protokolü ve kaynak hash'leri `protocol.json` içindedir.
