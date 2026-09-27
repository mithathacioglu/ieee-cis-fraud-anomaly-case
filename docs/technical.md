# Teknik kararlar

## Veri katmanı

### Left join ve anahtar sözleşmesi

Transaction tablosu ana evrendir. Identity'de kaydı olmayan işlem left join ile korunur. `validate="one_to_one"`, bir identity kaydındaki tekrarın işlemleri çoğaltmasını önler. Join öncesinde iki anahtarda da null/tekrar ve identity'de karşılıksız anahtar kontrol edilir. Beklenmeyen çakışan kolonlar sessiz suffix eklemek yerine hata üretir.

`has_identity`, sağ tablodaki kaydın varlığını temsil eder; sağdaki alanların doluluğunu değil. Bir identity satırındaki tüm değerler boş olsa da kaynak kaydı mevcutsa bu bayrak true'dur.

Ham veri değiştirilmez. CSV okumasındaki dtype korunur; profil aşamasında float32 downcast veya imputation yapılmaz. Parquet, birleştirilmiş veriyi tekrar CSV parse etmeden okumak için. Bu sürüm tüm tabloları belleğe alıyor; düşük RAM'li bir makinede çalışmaz.

### Şema çıkarımı

Kolonun fiziksel dtype'ı ve semantik rolü ayrı tutulur:

- `TransactionID`: join anahtarı.
- `isFraud`: değerlendirme etiketi.
- `TransactionDT`: elapsed time.
- `card*`, `addr*`, `M*`, `id_12`–`id_38`: kategori kodları için açık dataset override'ları.
- Diğer numeric alanlar: numeric; diğer string alanlar: categorical.
- Native datetime veya örneklemde doğrulanan ISO biçimli tarihler: datetime.

Datetime string tespiti en fazla 200 dolu değerle yapılan aday tespitidir; bütün kolonu dönüştürmez. Bu veri setindeki `TransactionDT` böyle bir aday değildir. Maskelenmiş numeric kolonların iş anlamı bilinmediğinde anlam uydurulmaz. Şema override'ları model seçimi veya fraud etiketi kullanmaz.

### Zaman ayrımı ve keşif kapsamı

İşlemler `TransactionDT`, sonra `TransactionID` ile sıralanır. İlk yaklaşık %60 train, sonraki %20 validation, kalan %20 test olur. Sınırda aynı `TransactionDT` değerine sahip bütün satırlar sonraki bölüme geçer. Bu nedenle tam satır yüzdeleri küçük farklar gösterebilir.

Temel dosya/join kalitesi tüm veride kontrol edilir. Keşif, outlier eşikleri, kategori frekansları ve ilişkiler train üzerinde hesaplanır. Validation/test dağılımlarına bakarak feature seçilmez. İnceleme eşiği train skorlarının üst %5'inden alındı, validation'da değil; validation'da seçilen şey context indirim gücüydü. Final test ayarlar sabitlendikten sonra bir kez çalıştırıldı. Gerçek bir sistemde gecikmeli etiketlerin ayrıca ele alınması gerekir, ancak bu veri dosyası etiketlerin ne zaman kullanılabilir olduğunu söylemiyor.

### Profil istatistikleri

Null oranı `missing / tüm satırlar`, benzersizlik oranı `distinct non-null / non-null` olarak tanımlanır. Sonsuz numeric değerler ayrıca sayılır ve dağılım hesabından çıkarılır. Core tutar ve zamanda sonlu olmayan değer varsa yükleme durur.

IQR outlier sınırları `Q1 - 1.5*IQR` ve `Q3 + 1.5*IQR`'dır. Bu bir fraud kararı değildir. IQR=0 dağılımlarda azınlık değerler bu ölçüte göre outlier olabilir; `zero_iqr` bayrağı bu sınırlamayı görünür kılar.

Kategori dağılımında ilk 10 değer, diğer dolu değerlerin toplamı ve eksik sayısı ayrı saklanır. Rare combination hesabı yalnızca bütün bileşenleri gözlenen satırlarda yapılır; destek <=5 inceleme eşiğidir. Eksik kombinasyon sayıları ayrı raporlanır.

Spearman korelasyonu, train'den seed=42 ile seçilen en fazla 30.000 satırda hesaplanır. Numeric, sabit olmayan ve null oranı <%95 olan kolonlar kullanılır. Her çift için en az 100 ortak gözlem aranır; gerçek destek sayısı rapora eklenir. Korelasyonları tek tek anlamlılık testi olarak yorumlamıyoruz; amaç olası tekrarları ve ilişkileri keşfetmek. Maskelenmiş V kolonlarında yüksek korelasyon görülmesi iş anlamını bildiğimiz anlamına gelmez.

### Entity vekili

`card1`, `card2`, `card3`, `card5`, `addr1` kombinasyonu kullanılır. Bir bileşen eksikse entity tanımsız kalır. Gerçek kişi kimliği olmadığı için aynı entity farklı kişileri birleştirebilir veya bir kişiyi birden fazla gruba bölebilir. Bu karar bir doğrulanmış kimlik çözümleme iddiası değildir.

Train içindeki işlem sayısı, tutar ortalaması/std, ilk/son görünme, ürün/e-posta alanı çeşitliliği ve ardışık işlem aralıkları raporlanır. `entity_profile.parquet` betimleyicidir; doğrudan model feature kaynağı olarak kullanılmaz. Feature engineering her işlem için geçmişi baştan hesaplıyor ve aynı saniyeli kayıtların birbirini görmediği ayrıca test ediliyor.

## Feature engineering

`FeatureBuilder` tek yazarlı bir geçmiş nesnesidir. Her `TransactionDT` grubu için iki aşama uygular: önce bütün satırların feature'ları mevcut geçmişten hesaplanır, sonra grubun işlemleri geçmişe eklenir. Aynı saniyeli işlemler birbirini göremez. Yeni batch önceki batch'in son zamanından sonra başlamalıdır; geç gelen veya aynı saniyeyi bölen batch reddedilir.

Entity geçmişinde count, Welford yöntemiyle tutar ortalaması/varyansı, son görünme zamanı, saat frekansları, ürün/e-posta alanı/cihaz açıklaması kümeleri tutulur. Velocity için [t-3600,t) ve [t-86400,t) aralıkları kullanılır. Welford güncellemesi önceki tüm tutarları saklamadan varyans hesaplar.

Global kategori ve kategori çifti frekanslarının paydası yalnızca daha önceki dolu gözlemlerdir. Eksik değerler ortak bir kategoriye çevrilmez. Yeni bir değer daha önce gözlenmiş bir kategori evreninde 0 frekans alabilir; hiç gözlem yoksa frekans bilinmiyor kalır.

`isFraud` feature oluşturma komutunda dosyadan okunmaz. `TransactionID` ve `split` output metadatasıdır; `FEATURE_SPECS` model girdi sözleşmesine dahil değildir. Feature'lar train, validation ve test boyunca sırayla üretilir. Validation/test döneminde yalnız etiketsiz geçmiş güncellenir; model parametreleri her durumda train'de fit edilir. Bu bir online geçmiş simülasyonu, dondurulmuş geçmiş deneyi değil.

`TransactionDT` için doğrulanmış başlangıç bilinmediğinden `hour`, `day_of_week`, `is_weekend`, `is_business_hours` varsayılan olarak null'dır; `calendar_available=0` bunu belirtir. Göreli gün ve 24 saat fazı yine hesaplanır. Explicit UTC offset içeren bir başlangıç verilirse takvim özellikleri bu sabit offset varsayımı altında doldurulur. Saat dilimi/DST çıkarımı yapılmaz. Bilinmeyen hafta sonunu false ile doldurmak desteklenmez.

Geçmişi olmayan ama tam entity anahtarlı işlemde count=0; eksik anahtarlı işlemde count=null'dır. Geçmiş ortalama yoksa, sıfırsa veya varyans yetersizse ilgili oran/z-score null kalır. En az beş önceki işlem geçmiş yeterliliği için başlangıç eşiğidir. Yirmi işlem ve yedi günlük tarihçe sık entity bayrağı üretir; bu güvenilir müşteri iddiası değildir. Bu eşikler henüz optimize edilmiş veya performansla doğrulanmış değildir.

DeviceInfo bir cihaz kimliği yerine cihaz açıklaması olabilir. E-posta alanı ilişkileri adres/kişi kimliği ilişkisi olarak yorumlanmaz.

## Dört anomali perspektifi

Column katmanı log tutarın train medyanından robust sapmasını kullanır. Multivariate katman 21 feature'ı açık bir allowlist'ten alır; kimlik, hedef, ham zaman ve toplam stream sayacı bu listede yoktur. Eksikler yalnızca train'de öğrenilmiş medyanla doldurulur ve missing indicator eklenir. Dolu gözlemi olmayan kolonların boyutu korunur. IsolationForest 128 ağaç ve ağaç başına en fazla 1024 örnekle, seed=42 ve n_jobs=1 olarak çalışır. `score_samples` yönü ters çevrilerek daha büyük değer daha anormal olacak hale getirilir.

Entity katmanı, en az 5 önceki işlem varsa tutarın kendi geçmişinden uzaklığını ve gözlenen ürün/e-posta alanı/cihaz açıklaması yeniliğini ayrı bileşenler olarak üretir. Tutar sapması paydası max(prior_std, 0.1*prior_mean, 1)'dir. İlişki yeniliğine 2 başlangıç puanı atanır; katman skoru bileşenlerin maksimumudur. Bunlar açıklanmış başlangıç sezgileridir, optimize edilmiş parametreler değildir.

Temporal katman aynı geçmiş koşulunda log1p(1h velocity), log1p(24h velocity) ve -log(entity_hour_probability) sinyallerinin train robust referanslarına göre pozitif sapmalarını kullanır. Saat nadirliği gerçek iş saati değil, göreli 24 saat fazı davranışıdır. Katman skoru üç bileşenin maksimumudur.

Yeterli geçmişi olmayan entity ve temporal skorlar null döner. Her katman ayrıca availability üretir. Açıklama çıktılarına gerçek gözlemler ve referanslar konur; IsolationForest için per-feature attribution hesaplanmadığından böyle bir açıklama iddia edilmez.

## Normalizasyon ve ağırlıklar

Her katman için train'deki sonlu raw skorlar sıralanır. Yeni skorun normalize değeri `count(reference < score) / N` olarak hesaplanır. Eşit değerlere strict lower rank uygulanır; özellikle sıfıra yığılmış temporal skorların yüksek yüzdelik alması önlenir. NaN normalize edilirken de NaN kalır. Infinity reddedilir.

Ağırlıkları dörtte bir, eşit bıraktım. Bir katmanın diğerinden daha iyi fraud yakaladığına dair elimde ölçüm yoktu; kanıtsız üstünlük vermedim. Sadece mevcut katmanların ağırlıkları yeniden normalize edilir. `available_weight` toplam bilgi kapsamasını; her katmanın etkin ağırlığı ve contribution alanı final skora katkısını gösterir. Mevcut pozitif ağırlık yoksa hata üretilir. Bu skor fraud olasılığı değil. Dört katmanı da olan işlemlerle eksik katmanlı işlemlerin eşik davranışı ayrı ayrı ölçüldü; sonuçlar `artifacts/context/evaluation.json` içindeki `audit_by_coverage` alanında.

Ağırlıkları değiştirmeyi sonradan denedim: dokuz kombinasyon ve üç context politikası adayı, validation'ın iki kronolojik bloğunda aynı %5 bütçeyle. Hiçbiri eşit ağırlığı geçemedi, çalışan yapılandırma değişmedi. Aday listeleri ve blok metrikleri `artifacts/scoring_development/` ile `artifacts/context_review/` altında.

### Gözetimli modellerle karşılaştırma

Dört anomali katmanı etiket okumuyor. İlk karşılaştırmada iki gözetimli model train üzerinde eğitilip sonraki validation yarısında değerlendirildi: aynı 2.953 incelemede ham skor 230, logistic regression 597, gradient boosting 628 fraud yakaladı. AUC değerleri ham skor için 0,673, gradient boosting için 0,796.

Gradient boosting'de `early_stopping` kapalı. Varsayılan `"auto"` bu boyutta veride sklearn tarafından açılır ve eğitim setinden **rastgele** %10 iç validation ayırır; kronolojik split'e bu kadar dikkat edilen bir yerde tutarsız olurdu. Kapalı tutma gerekçem protokol: iç bölme rastgele olursa modelin ne zaman durduğu kronolojik olmayan bir kesite bağlanır.

Açıkken aynı koşu 641, kapalıyken 628 TP veriyor. Bu 13 farkı tek bir nedene bağlamıyorum. `auto` iki şeyi birlikte değiştiriyor: eğitimde kullanılan satır sayısı %10 azalıyor ve iterasyon sayısı 100'de sabit kalmak yerine erken kesiliyor. Bu koşu rastgele bölmenin payını diğerinden ayırmıyor, ayırmak için bölme türünü sabit tutup yalnızca iterasyon sayısını değiştiren ayrı bir koşu gerekir; onu yapmadım. İç bölmenin tamamı dış eğitim döneminin içinde kaldığı için bu, tek başına dış değerlendirme verisine sızıntı da değildi. Yani manşet sayıyı düşüren değişikliği doğru bulmamın nedeni farkın yönü değil, bölmenin kronolojik olması.

400 tekrarlı eşlenmiş satır bootstrap'ında gradient boosting'in TP farkı [348, 444] aralığındaydı. Bu hesap mevcut modellere ve değerlendirme dönemine koşulludur; entity ve zaman bağımlılığını korumaz. Sıfırı dışlaması, farkın gelecek dönemlerde korunacağını göstermez. Bağımlılık hesaba katıldığında aralığın ne yönde ve ne kadar değişeceği bu hesapla bilinmez.

Zaman duyarlılığı koşusunda `prior_global_count` ve `relative_day` birlikte çıkarıldı. Gradient boosting'de %5 bütçedeki TP 628'den 593'e, %10'da 943'ten 919'a indi; buna karşılık %1 bütçede 191'den 216'ya çıktı ve AP 0,1640'tan 0,1663'e yükseldi. Bu sonuçlara bir eşdeğerlik veya anlamlılık testi uygulanmadı. Diğer geçmiş feature'ları da zaman bilgisi taşıyabilir; iki kolonu çıkarmak zaman etkisini bütünüyle izole etmez. Zaman bilgisinin bulunması tek başına sızıntı değildir; kritik koşul, girdinin karar anında erişilebilir olmasıdır.

İlk karşılaştırma **girdi bakımından eşlenmiş değil.** Gözetimli modeller 36 feature, dört anomali katmanı toplam 26, IsolationForest 21 feature kullanıyor. Girdi farkı sonuçların bir parçası. İki gözetimli modelde de `balanced` sınıf ağırlığı kullanıldı; aynı ağırlık seçimi algoritmaların diğer farklarını ortadan kaldırmaz.

Ek [kronolojik deneyde](../reports/temporal_validation.md) modeller üç ayrı dönem öncesinde yeniden eğitilir; tam, iki zaman kolonu çıkarılmış ve eşlenmiş girdi kümeleri karşılaştırılır. 0 ve 7 günlük etiket kesinleşme varsayımları eğitim kesimini değiştirir. Bunlar veri setindeki işlem zamanına göredir; proje hazırlama veya teslim süresiyle ilgili değildir. Gerçek etiket kesinleşme zamanı veri setinde bulunmaz. Belirsizlik için eşlenmiş 6 ve 24 saatlik zaman blokları kullanılır. [Protokol](validation_protocol.md), korunabilen bağımlılıkları ve kalan varsayımları açıklar.

Gözetimsiz motorun etiketsiz skor üretmesi yeni fraud desenlerini yakaladığını kanıtlamaz. Hibrit kullanımın faydasını ölçmek için sabit %50 kotalı inceleme kuyruğu aynı bütçedeki gözetimli sıralamayla karşılaştırılır. Sonuç üretim entegrasyonu kararı değildir; maliyet ve yeni dönem doğrulaması ayrıca gerekir.

Fit ve skor üretimi `isFraud` okumaz. Model ve kalibrasyon referansları train'de sabittir; değişen tek kısım önceden hesaplanmış kronolojik etiketsiz geçmiş feature'larıdır. İnceleme eşiği train skorlarının üst %5'inden alındı, yani etiket görmeden sabitlendi; validation'da etiketle seçilen şey context indirim gücüydü. Ağırlıklar hiç seçilmedi, eşit bırakıldı. Final test etiketleri hiçbir tasarım kararına girmedi.

41 feature'ı bütün veride ürettim; geçmiş ve velocity hesaplarını bağımsız bir yöntemle karşılaştırdım. Kaydettiğim modeli yeniden yükleyip 128 satırın skorunun aynen çıktığını ve katman katkılarının her satırda final skora eşit olduğunu kontrol ettim. Sayısal sonuçlar `reports/features.md` ve `reports/anomaly_scores.md` içinde.

## Context adjustment

Altı context hipotezi ayrı katman katkılarını azaltır: ürün türünde alışıldık tutar, sık/alışıldık entity faaliyeti, entity için alışıldık yüksek tutar, iş saatleri, beklenen hafta sonu çalışması ve doğrulanmış dış güven kaydı. İndirim aynı katmanda en fazla bir kuraldan gelir (en büyük öneri); eşitlikte tanımlama sırası korunur. Toplam azaltım en fazla 0.05 puan ve raw skorun %15'idir. Böylece bütün skoru tek çarpanla düşürmek yerine hangi sinyalin bağlamla yeniden yorumlandığı izlenir.

İlk üç hipotez kaynak verinin izin verdiği kadarıyla çalışır. Ürün referansı en az 1000 train gözlemi olan ProductCD grubunun p10–p90 tutar aralığıdır; kodlara iş anlamı atfedilmez. Bu aralıkta ve global tutar rank >=0.90 ise column katkısına sınırlı indirim önerilir. Bilinmeyen ürün veya eksik referans indirim üretmez.

Güçlü entity sapması (raw >=3), multivariate rank >=0.98, saatlik en az 10 işlem veya bilinen yeni ilişki indirimi bloke eder. Entity temelli indirimler ayrıca en az 20 geçmiş işlem, alışıldık tutar, bilinen ürün ilişkisi ve düşük entity sapması ister. Frequent bayrağının yedi günlük gözlem koşulu korunur. Bunlar operasyonel hipotezlerdir; şirketin sağladığı politikalar değildir.

İş saatleri ve hafta sonu kuralları ancak açık takvim özellikleri varsa çalışır. Hafta sonu faaliyeti ayrıca önceden gözlenmiş dış program ister. Trusted entity indirimi, işlemden önceki trust_observed_at ve boş olmayan trust_source olmadan çalışmaz. Same-second veya gelecekteki kayıt kabul edilmez. Sık işlem geçmişinden güven etiketi türetilmez. Sağlanan kaynağın kimliğini doğrulayan harici servis bu prototipte yoktur; zaman/kaynak sözleşmesi kontrol edilir.

Eşik olarak train raw skorlarının %95 yüzdeliğini, yani 0.85167813'ü sabitledim. %5 inceleme kapasitesi ve en fazla 1 yüzde puan recall kaybı benim deney varsayımlarım; kimse bana böyle bir hedef vermedi. Güç adaylarını (0 / 0.25 / 0.5 / 1) etiketlere bakmadan önce belirledim, seçimi yalnız validation'ın ilk yarısında yaptım: recall sınırını aşmayanlar arasında en çok FP azaltan, eşitlikte daha az recall kaybettiren, sonra daha küçük olan. 0.5 çıktı.

Validation'ın ikinci yarısı bir geliştirme kontrolü; bu bölümün toplamlarını geliştirme sırasında zaten görmüştüm, yani hiç açılmamış holdout değil.

Sonuç: FP 115 azaldı (%5,65), ama TP de 17 azaldı; bu, baseline'ın yakaladığı fraud'ların %9,88'i. Recall kaybı 0,825 yüzde puan. Aynı inceleme kapasitesinde 2 fraud daha az yakalandı. İki sayıyı birlikte okumak gerek: 0,825 puanlık kayıp küçük görünüyor ama yakaladıklarımın onda birini kaybetmişim. Sıralamanın iyileştiğini söyleyemem.

Maliyet verisi verilmediği için kararı oran olarak yazdım: 115/17, yani başa baş noktası 6,8. Context ancak kaçırılan bir fraud, 6,8 yanlış alarmdan ucuzsa kazandırır. Aynı aritmetiği final testin sayılarına uyguladığımda 136/28 = 4,9 çıkıyor, yani orada koşul daha da zor. İki dönemde de bu koşulun sağlanmasını beklemiyorum: bir yanlış alarm birkaç dakikalık inceleme, kaçırılan fraud işlem tutarı artı geri ödeme. Bu yüzden servisin varsayılan profilini indirim uygulamayacak şekilde bıraktım; kararın kendisi şirketin iki maliyetini bu eşiğe koymasıyla verilir.

6. adımın ölçüm/trace çıktıları `artifacts/context` ve `reports/context_adjustment.md` altında tutulur. Her işlemde rule match, guard, katman indirim tutarı, kazanan neden, düzeltilmiş katkı ve final adjusted skor kaydedilir. Testler eksik/future güven kayıtlarını, caps, aynı katmanda çakışmayı, doğru FPR paydasını ve deterministik bütçe eşitlik çözümünü kapsar. Bu adım sonunda test sayısı 64'tür.

## Configurable rule engine

`RuleEngine` JSON'daki koşul ağacını yorumlar. `all` ve `any` grupları; eq/ne/gt/gte/lt/lte/in/is_missing/is_present yaprakları desteklenir. Çalıştırılabilir kod veya Python eval kullanılmaz. İzin verilen alanlar feature sözleşmesi ve belirli skor/context alanlarıdır; hedef, split ve TransactionID koşul girdisi olamaz. Yapılandırma sürümü, anahtarlar, tekil ID'ler, literal türleri, priority ve action başlangıçta doğrulanır. En fazla 100 kural, kural başına 100 düğüm, grup başına 20 çocuk ve altı seviye derinlik kabul edilir.

Eksik kolon sessizce eşleşmemezliğe dönüşmez; çağrı hata verir. Eksik hücreler normal karşılaştırmalarda eşleşmez; özellikle NaN != değer true kabul edilmez. Eksiklik için açık is_missing vardır. Bütün değerler sonlu sayısal/boolean veya null olmalıdır; gerekli normalize skorlar [0,1] aralığında doğrulanır. ID/index tekilliği batch hizasını korur.

Motor yüksek priority, sonra alfabetik rule ID sırasını uygular. İlk eşleşen kural karar verir; kalan eşleşmeler saklanır. Review/monitor birlikte önerildiyse action_conflict işaretlenir. Varsayılan no_rule_match'tir. Bu çıktılar finansal işlem üzerinde bloklama veya onay çalıştırmaz; inceleme önerisidir. Mevcut 10 kurallı politikada tüm review öncelikleri monitor'dan yüksektir.

Policy dosyası doğrulandıktan sonra kopyalanır. `config` özelliği de kopya döndürür; dış nesne değişikliği çalışan motoru değiştirmez. Yeni JSON ile yeni motor örneği yaratılarak kod değişmeden politika güncellenir. Dosyayı otomatik izleyen veya istek ortasında değiştiren mekanizma yoktur. Canonical JSON SHA-256 açıklamalarda bulunur; orijinal dosya hash'i ve son çalıştırılan snapshot ayrıca saklanır.

Skorlar değiştirilmediği için upstream anomali ve downstream iş kararı ayrı incelenebilir. Açıklama her koşulda eşik, gözlenen değer, eksiklik ve boolean sonuç; her kuralda selected/superseded/not_matched/disabled durumu üretir. Açıklama şablonu gözlem ağacıyla desteklenir. Bu çıktı RAG katmanına kanıt olarak taşındı; `explanations.py` kararı, kazanan kuralı ve gözlenen değerleri oradan yazıyor.

Kuralların 590.540 işlem replay'i etiket okumaz. Değerlendirme yalnızca zaten görülmüş sonraki validation bölümünün etiketlerini okur. Eşikler bu sonuçlara göre ayarlanmadı. Aynı 59.054 işlemde context>=0.85 politikası 2.135 inceleme/162 TP/1.973 FP; kural politikası 3.667 inceleme/236 TP/3.431 FP üretti. Kuralların daha fazla fraud yakalaması daha fazla iş yükü ve daha düşük precision ile birlikte gelir. Kurallar için ek ranking skoru tanımlanmadığından aynı-bütçe sıralama üstünlüğü iddia edilmez. 44 kural/karar metriği testiyle toplam 108 test geçer. `scripts/verify_rules.py` kayıtlı kararları bağımsız scalar yorumlayıcıyla örneklem üzerinde karşılaştırır.

## RAG, ajanlar ve servis

### Sonraki ürün riski context adayı

Case'in işlem tipi risk profili önerisi için `ProductRiskContextEngine` ayrı bir adaydır. Dört anomali katmanı, train referansları ve eşit ağırlıklar değişmez. Ürünlerin train fraud sayımları genel orana 1.000 sanal gözlemle yumuşatılır; bu bağlam katmanı gözetimlidir. En az 1.000 gözlem destek koşuludur. Sınırlı log risk oranı × validation'da seçilen güç, davranış sonrası skora eklenir; güçlü anomali koruması negatif eklemeyi engeller. Son skor [0,1] ile sınırlıdır ve olasılık sayılmaz.

Bu adayda eski ürün-olağan-tutar indirimi kaldırılır; iş saati/hafta sonu/güven ve diğer davranış kuralları korunur. Ürün riski ayrı additive terimle açıklanır; katmanlara nedensel attribution uydurulmaz. Train kesiminden eski/eşit işlemlerde profil devre dışıdır. Fraud etiketlerinin train dönemi sonunda ulaşılabilir olduğu varsayılır; bildirim gecikmesi sağlanmamıştır.

[Geliştirme deneyi](../reports/product_context.md) önceki final test açıldıktan sonra, yalnızca train/validation etiketleriyle yapılmıştır. Aynı %5 bütçede TP 230 -> 242 ve FP 2.723 -> 2.711; %1 bütçede TP 46 -> 37 olmuştur.

12 işlemlik farkın belirsizliğini ayrıca ölçtüm, çünkü bu kadar küçük bir farkı ölçmeden savunmak istemedim. Kronolojik deneydeki eşlenmiş blok bootstrap'ının aynısını bu karşılaştırmaya uyguladım ([belirsizlik raporu](../reports/product_uncertainty.md)). %5 kapasitede aralık 24 saatlik blokla [-6,9, +47,1] ve 6 saatlikle [-10,6, +44,7]; ikisi de sıfırı içeriyor. %10 kapasitede +94 farkın aralığı [27,9, 169,0] ile sıfırı dışlıyor ve audit döneminin iki yarısında da aynı yönde. %1 kapasitede fark negatif ve yön dönem değiştirince dönüyor. Yani bu katmanın değeri kapasite politikasına bağlı; kapasite sabitlenmeden karar verilmemeli. Bu nedenle varsayılan profil `raw` kalır; `FRAUD_CONTEXT_PROFILE=product_risk` ile karşılaştırma yapılır. Sonraki validation daha önce görüldüğü için yeni bağımsız holdout iddiası yoktur. [Kural etkisi](../reports/product_profile_verification.md) farklı alarm bütçelerini açıkça gösterir.

Yerel embedding/RAG, ajan görev devri, FastAPI ve dependency_injector container uygulanmıştır. [Runtime mimarisi](runtime.md) veri akışını, provider ömürlerini, mesaj sözleşmesini, offline kurulumu ve API sınırlarını açıklar. Gerçek model yanıtları [RAG raporundadır](../reports/local_rag.md).

Nihai değerlendirme tamamlandı. Yukarıdaki 44 / 64 / 108 test sayıları ilgili adımın bittiği andaki durumu gösterir; güncel toplam [doğrulama özetindedir](../reports/verification_summary.json). Sabit model/context/rule parametreleriyle alınan [final test sonucu](../reports/final_evaluation.md) context'te sıralama iyileşmesi göstermedi. Etiketleri gördükten sonra risk modeli veya eşik ayarlanmadı; LLM açıklama formatındaki düzeltmeler risk hesabına dokunmuyor.

Üretim için kalan konular: tam ve kalıcı causal history servisi, kimlik doğrulama, yetkilendirme, işlem kuyruğu, politika yayınlama süreci, yük testleri, drift izleme ve gerçek iş maliyetlerine dayalı eşik seçimi.

## Kaynaklar

- [IEEE-CIS veri açıklaması](https://www.kaggle.com/competitions/ieee-fraud-detection/data): dosyalar ve `TransactionDT` semantiği.
- [Kaggle CLI](https://github.com/Kaggle/kaggle-cli): resmî indirme ve kimlik doğrulama aracı.
