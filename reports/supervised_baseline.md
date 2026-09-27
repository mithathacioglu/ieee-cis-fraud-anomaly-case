# Gözetimli baseline karşılaştırması

İki gözetimli model, dört anomali katmanının birleşik skoruyla aynı değerlendirme döneminde karşılaştırıldı.

Aynı kronolojik split. Takvim referansı verilmediği için train'de tamamen boş veya sabit kalan 5 kolon elendi (`calendar_available`, `day_of_week`, `hour`, `is_business_hours`, `is_weekend`); gözetimli modeller kalan 36 feature'ı görüyor. Karşılaştırma sonraki validation yarısının 59,054 satırında, 2,061 gerçek fraud üzerinde yapıldı. İki gözetimli model de aynı sınıf ağırlığıyla eğitildi. Final test bölümü açılmadı.

**Girdiler eşlenmiş değil.** Gözetimli modeller 36 feature görüyor; dört anomali katmanının toplam girdisi 26, IsolationForest'ınki 21. Yani aşağıdaki fark yalnız etiket kullanmaktan gelmiyor, girdi kümesi farkı da içinde. Kontrollü bir deney değil, bir referans noktası.

| Yöntem | Etiket ister | AP | ROC AUC |
|---|---|---:|---:|
| Gözetimsiz raw skor | hayır | 0.0634 | 0.6731 |
| Logistic regression | evet | 0.1497 | 0.7488 |
| Gradient boosting | evet | 0.1640 | 0.7964 |

## Aynı inceleme bütçesinde yakalanan fraud

| Bütçe | İnceleme | Gözetimsiz raw skor | Logistic regression | Gradient boosting |
|---|---:|---:|---:|---:|
| %1 | 591 | 46 | 205 | 191 |
| %5 | 2,953 | 230 | 597 | 628 |
| %10 | 5,906 | 457 | 849 | 943 |

## Eşlenmiş satır bootstrap aralığı

%5 bütçedeki TP farkı, her iki model için aynı satırlar 400 kez yeniden örneklenerek hesaplandı. Modeller yeniden eğitilmedi. Bu aralık mevcut döneme ve eğitilmiş modellere koşulludur. Satırların bağımsız olduğu varsayımı entity ve zaman bağımlılığını korumaz; aralığın nominal %95 kapsamı garanti değildir. Hatanın yönü ve büyüklüğü burada ölçülmedi. Sıfırı dışlaması, farkın gelecek dönemlerde korunacağını göstermez.

| Model | Gözlenen fark | %95 aralık | Sıfırı dışlıyor mu |
|---|---:|---:|---|
| Logistic regression | +367 | [315, 411] | evet |
| Gradient boosting | +398 | [348, 444] | evet |

## Zaman taşıyan feature'lar çıkarılınca

Çıkarılan kolonlar: `prior_global_count`, `relative_day`. Bu kolonlar zamanın ilerlemesini taşır. Bunların kullanılması tek başına sızıntı anlamına gelmez; ablation, bu girdilere duyarlılığı ölçer. Aşağıdaki değerler aynı değerlendirme dönemine aittir:

| Model | AP | ROC AUC | TP %1 | TP %5 | TP %10 | Tam kümeye göre |
|---|---:|---:|---:|---:|---:|---|
| Logistic regression | 0.1437 | 0.7572 | 172 | 598 | 845 | AP düştü, AUC yükseldi, %1 -33, %5 +1, %10 -4 |
| Gradient boosting | 0.1663 | 0.7862 | 216 | 593 | 919 | AP yükseldi, AUC düştü, %1 +25, %5 -35, %10 -24 |

Logistic regression: metrikler farklı yönlerde değişti (2 artış, 3 düşüş, 0 eşitlik).

Gradient boosting: metrikler farklı yönlerde değişti (2 artış, 3 düşüş, 0 eşitlik).

Bu nokta tahminleri anlamlılık veya eşdeğerlik testi değildir. Zaman bilgisi diğer geçmiş feature'larında da bulunabilir. Bu ablation sızıntıyı dışlamaz ve zaman etkisini bütünüyle izole etmez. Dönemler arası tutarlılık ayrıca kronolojik değerlendirmeyle ölçülmelidir.

## Yorum

Aynı 2,953 incelemede gözetimsiz skor 230 fraud yakalarken gradient boosting 628 yakalıyor. Oran 2.7. AUC farkı 0.123.

Girdi kümeleri farklı olduğundan bu fark yalnız etiket kullanımına atfedilemez. Ham veri kolonlarının tamamıyla ayrı bir model araması yapılmadı; bu sonuç bir performans tavanı değildir.

Gözetimli modeller eğitim etiketlerinin hazır olduğunu varsayar. Veri etiketin ne zaman kesinleştiğini içermediğinden bu varsayımın üretimdeki karşılığı bilinmiyor. Gözetimsiz motor etiket olmadan skor üretebilir; bu, yeni fraud desenlerini yakaladığının kanıtı değildir.

Hibrit kullanımın ek faydası bu tabloda ölçülmedi. Bunun için aynı toplam inceleme bütçesinde birleşik kararın ve yalnız gözetimli modelin karşılaştırılması gerekir.

Sayıların kaynağı: `artifacts/baseline/evaluation.json`. Yeniden çalıştırma: `python scripts/supervised_baseline.py`.
