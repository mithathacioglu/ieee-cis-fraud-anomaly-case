# Kronolojik değerlendirme

Bu çalışma, mevcut tek dönem karşılaştırmasının hangi koşullarda değiştiğini ölçer. Case'in zorunlu on adımına ek bir analizdir; servis akışını değiştirmez. Train ve validation daha önce incelendiği için sonuçlar geliştirme sonuçlarıdır. Final test bu çalışmada kullanılmaz.

## Deney düzeni

- Validation zaman aralığı üç eşit süreli pencereye ayrılır. Aynı saniyedeki işlemler aynı pencerede kalır. Pencerelerin işlem sayıları eşit olmak zorunda değildir.
- Her pencere öncesinde modeller yeniden eğitilir. Eğitim aralığı genişler; sonraki pencere eğitiminde önceki validation işlemleri kullanılabilir.
- Etiketlerin işlemden 0 veya 7 gün sonra hazır olduğu iki ayrı senaryo uygulanır. Gerçek etiket kesinleşme zamanı veri setinde yoktur; bu süreler ölçülmüş gecikmeler değildir. Pozitif ve negatif etiketler için aynı gecikme varsayılır.
- Karşılaştırılan bütün modeller aynı eğitim satırlarını ve aynı değerlendirme işlemlerini kullanır. Gecikme uygulanınca anomali motorunun eğitimi de aynı tarihte kesilir. Bu, eğitim dönemi farkını kontrol eder; anomali motorunun daha güncel etiketsiz veriyle eğitilebildiği üretim seçeneğini ölçmez.
- Geçmiş feature'ları kronolojik ve etiketsiz akıştan gelir. Değerlendirme döneminde daha önce gerçekleşmiş işlemler geçmişi günceller; aynı saniyedekiler birbirini görmez. Eğitim kesimi ile değerlendirme arasındaki işlemler de etiketsiz geçmişte bulunabilir. Bu, çevrim içi geçmiş varsayımıdır.

## Girdi kümeleri

Her eğitim penceresinde üç gözetimli koşu vardır:

1. Tam küme: yalnız eğitimde tamamen boş veya sabit kolonlar çıkarılır.
2. Zaman ablation'ı: tam kümeden `prior_global_count` ve `relative_day` çıkarılır.
3. Eşlenmiş küme: anomali motorunun kullandığı 26 kaynak feature kullanılır.

Eşlenmiş küme girdi farkını kontrol eder. Algoritma, kayıp fonksiyonu, dönüşümler ve etiket kullanımı hâlâ farklıdır; kalan fark yalnız etiketlerin nedensel etkisi olarak yorumlanmaz. İki zaman kolonunun çıkarılması diğer geçmiş değişkenlerindeki zaman bilgisini ortadan kaldırmaz.

Logistic regression ve histogram gradient boosting, `balanced` sınıf ağırlığıyla eğitilir. İmputer ve scaler her pencerenin eğitim bölümünde fit edilir. Gradient boosting'de rastgele iç validation bölünmesini önlemek için early stopping kapalıdır; 100 iterasyon sabittir. Anomali motoru ve skor kalibrasyonu her pencerede yeniden fit edilir. Hiperparametre araması yapılmaz. Bu nedenle yeni skorların eski tek dönem raporuyla birebir aynı olması beklenmez.

## Bütçe ve hibrit karar

Birincil ölçüt %5 inceleme bütçesindeki TP sayısıdır; %1 ve %10 da birlikte raporlanır. Her pencerede bütçe `ceil(işlem sayısı × oran)` olur. Eşit skorda küçük TransactionID önce gelir. AP, ROC AUC, precision, recall, FP ve FN yardımcı ölçütlerdir.

Hibrit adayda inceleme sıralaması eşlenmiş gradient boosting ile ham anomali sıralaması arasında sabit %50 kota ile oluşturulur. Bir işlem ikinci kez seçilmez; tekrar karşılaşılırsa ilgili listenin sıradaki işlemi alınır. Kota sonuçlara bakılarak ayarlanmaz. Böylece hibrit ile gözetimli model aynı toplam inceleme sayısında karşılaştırılır. Bu bir toplu inceleme kuyruğu deneyidir; çevrim içi eşik politikası veya kalibre edilmiş fraud olasılığı değildir.

İki modelin ayrı listelerindeki ortak ve farklı fraud yakalamaları da kaydedilir. Listeleri birleştirmek bütçeyi büyütür; birleşimin TP sayısı aynı bütçedeki kazanım gibi sunulmaz.

## Belirsizlik

Her pencerede 6 ve 24 saatlik ardışık, çakışmayan zaman blokları kullanılır. Bloklar ilk değerlendirme işleminden başlar. Boş olmayan bloklar yerine koyarak örneklenir; blok içindeki tüm işlemler birlikte taşınır. Karşılaştırılan sıralamalarda aynı örnek kullanılır. En az sekiz blok yoksa aralık üretilmez.

Blokların işlem sayıları farklı olduğu için her örnekte %5 bütçe yeniden hesaplanır. TP farkı işlem sayısına bölünüp özgün pencerenin işlem sayısıyla çarpılır. Raporlanan yüzdelik aralık bu ölçektedir; örnek başına ham TP farklarının aralığı değildir. Her blok uzunluğu için 1.000 tekrar yapılır; daha elverişli görünen aralık seçilmez.

Bu yöntem blok içindeki bağımlılığı korur. Blokların değiştirilebilir olduğu varsayımını gerektirir; farklı bloklarda görülen aynı entity'nin bağımlılığını ve modelin yeniden eğitilmesindeki belirsizliği kapsamaz. Aralıklar mevcut eğitilmiş modellere ve döneme koşulludur. Hibrit kuyruğa bootstrap aralığı verilmez: kuyruk tahsisinin her örnekte yeniden kurulması gerekir. Hibritin nokta sonuçları ayrı dönemlerde karşılaştırılır.

Pencereler arasındaki değişim ayrıca gösterilir. Eğitim bölümleri örtüştüğü için üç fold bağımsız tekrar sayılmaz; fold ortalamasına bağımsız örneklem standart hatası uygulanmaz. Birden fazla model, gecikme ve blok uzunluğu incelendiğinden aralıklar keşifseldir; düzeltilmiş çoklu hipotez testi yapılmaz.

## Çalıştırma ve kayıt

Protokol [config/temporal_validation.json](../config/temporal_validation.json) içinde sabittir. Betik koşu başında protokolü, sürümleri ve kaynak kod hash'lerini yazar. Tahminler ve pencere sonuçları ayrı dosyalara kaydedilir. Var olan çıktı klasörünün üzerine yazılmaz.

```powershell
python scripts/temporal_validation.py --data-root D:/Proje/data/processed --output artifacts/temporal_validation
```

Fraud etiketi kesinleşme zamanı ve gerçek operasyon maliyetleri olmadığından bu deney üretim kârlılığı hesaplamaz. %5 bütçe bir kapasite varsayımıdır. Üretim kararı için etiket gecikmesi, yanlış alarm maliyeti, kaçırılan fraud kaybı ve yeni dönem doğrulaması gerekir.

Yöntem referansları: [scikit-learn: kronolojik çapraz doğrulama](https://scikit-learn.org/stable/modules/cross_validation.html#time-series-split), [SciPy: eşlenmiş bootstrap örneklemesi](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bootstrap.html). Buradaki zaman blokları özel uygulanır; SciPy'nin satır bootstrap'ı doğrudan kullanılmaz.
