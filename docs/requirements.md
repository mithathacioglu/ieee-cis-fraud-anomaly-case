# Case adımları ve karşılıkları

Kaynak: `LOGO_Data_Science_Case_Study.pdf`, 5 sayfa. Aşağıda PDF'in on adımını sırayla ele alıyorum: ne yaptım, hangi sayıyı ölçtüm, nerede duruyor. Sonuç lehime değilse de öyle yazdım, özellikle 6. adımda.

Metindeki aşama test sayıları o adımın bittiği andaki durumu gösterir. Önceki teslimin kontrolleri [doğrulama özetinde](../reports/verification_summary.json); ek deney testleriyle güncel yerel toplam README'de bulunur.

## Ön hazırlık: veri

`scripts/download_data.py` Kaggle'dan yalnız `train_transaction.csv` ve `train_identity.csv` indiriyor. Betik ZIP'i açarken bütünlüğü, sonra CSV'de kolon sayısını, TransactionID tekilliğini, etiket değerlerini ve identity referanslarını kontrol ediyor. Dosya özetleri `data/raw/manifest.json` içine yazılıyor; bunlar hangi sürümle çalıştığımı görmek için, veri sağlayıcının imzası değil.

## Adım 1: Yükleme ve birleştirme

İki tabloyu `data.merge_tables` ile TransactionID üzerinden birleştirdim: left join, `validate="one_to_one"`. Left join'i seçtim çünkü inner join identity kaydı olmayan 446.307 işlemi siler; yani tespit etmeye çalıştığım işlemlerin çoğunu modele hiç sokmaz.

590.540 işlem ve 144.233 identity kaydı, birleştirmede tek satır kaybı yok. Identity eşleşmesi %24,42. `has_identity` bayrağı identity **kaydının** varlığını gösteriyor, alanların dolu olmasını değil; ikisi farklı şey ve ayrı tutuluyor.

Eksik veri, kolon tipleri, kalite ve dağılımlar `schema.py` ile `profiling.py` içinde. Çıktılar [veri raporunda](../reports/data_profile.md). Veri sözleşmesi, zaman sınırı ve entity eksikliği için 14 test var.

## Adım 2: Profiling

Bu adımda ayırmaya çalıştığım şey şu: bir kolonun fiziksel dtype'ı ile semantik rolü aynı şey değil. `card1` sayı olarak saklanıyor ama kategori kodu; iki kart numarası arasındaki fark bir mesafe ifade etmiyor. `schema.py` bu yüzden `card*`, `addr*`, `M*` ve `id_12`–`id_38` için açık override taşıyor. `TransactionDT` de ayrı bir rol aldı: takvim tarihi değil, açıklanmamış bir referanstan geçen saniye.

Ölçtüklerim: null oranları, IQR ve quantile'lar, kategori frekansları, üç kategori çifti için nadir kombinasyonlar, train'den 30.000 satırlık örneklemde Spearman (her çift için en az 100 ortak gözlem) ve entity davranış özetleri.

Entity dediğim şey `card1+card2+card3+card5+addr1` kombinasyonu; gerçek müşteri kimliği değil. Aynı anahtar farklı kişileri birleştirebilir, bir kişi farklı anahtarlara dağılabilir. Bileşeni eksik olan işlemleri tek bir "bilinmeyen" grubuna toplamadım; öyle yapsam alakasız satırlar aynı geçmişi paylaşırdı.

## Adım 3: Feature engineering

41 özellik ürettim, dördü de PDF'in istediği gruplarda: temporal, entity, relational, context. Tamamı 590.540 satır için hesaplandı.

Buradaki asıl mesele sızıntı. `FeatureBuilder` her zaman grubunda önce bütün satırların özelliklerini mevcut geçmişten hesaplıyor, **sonra** grubu geçmişe ekliyor. Yani aynı saniyedeki iki işlem birbirini görmüyor; kaynağın zaman çözünürlüğü hangisinin önce geldiğini söylemiyor, ben de uydurmadım. Batch sınırında da aynı kural geçerli: aynı saniye ikiye bölünemiyor.

`isFraud` bu komutta dosyadan okunmuyor bile; `build_features.py` parquet'i yalnız gerekli kolonlarla açıyor.

Kendi hesabıma güvenmemek için `scripts/verify_features.py` yazdım: aynı sayıları tamamen farklı bir yöntemle, numpy `searchsorted` ve `groupby().cumsum()` ile yeniden üretiyor. 514.655 tam entity anahtarlı işlemde geçmiş count/mean ve 1/24 saat velocity hesapları tutuyor.

Takvim referansı verilmezse `hour`, `day_of_week`, `is_weekend` ve `is_business_hours` null kalıyor. IEEE-CIS başlangıç tarihi vermiyor, ben de bir tarih varsayıp saat üretmedim. `--calendar-reference` ile açık bir tarih verilebiliyor, ama o tarih çağıranın varsayımı.

18 feature testi var.

## Adım 4: Dört anomali katmanı

`anomaly.py` dört ayrı raw skor üretiyor ve bunlar gerçekten farklı yöntemler, aynı hesabın varyasyonu değil:

- **Column:** log tutarın train medyanından robust sapması.
- **Multivariate:** 21 feature'lık açık bir allowlist üzerinde IsolationForest. Kimlik, hedef, ham zaman ve stream sayacı bu listede yok.
- **Entity:** max(tutarın kendi geçmişinden sapması, gözlenen ilişki yeniliği).
- **Temporal:** max(1 saat velocity, 24 saat velocity, entity saat-fazı nadirliği).

Entity ve temporal katmanlar en az 5 geçmiş işlem istiyor; yoksa null dönüyor. Bunu sıfır saymadım; geçmişi olmayan bir işlemi "daha güvenli" göstermek yanlış olurdu.

IsolationForest için feature attribution hesaplamadım. Gözlenen girdileri ve toplam skoru raporluyorum; uydurma bir SHAP açıklaması üretmiyorum. PDF belirli bir attribution yöntemi şart koşmuyor.

## Adım 5: Normalizasyon ve birleştirme

Her katmanı kendi train skorlarının sıralı referansına göre normalize ediyorum: `count(train_raw < skor) / N`. Strict lower rank seçtim çünkü temporal skorlar sıfıra yığılıyor; normal rank kullansam o kalabalık grup yüksek bir yüzdelik alırdı.

Ağırlıklar dörtte bir, eşit. Bir katmanın diğerinden daha iyi fraud yakaladığına dair ölçümüm yok, o yüzden kanıtsız üstünlük vermedim. Sonradan dokuz farklı ağırlık kombinasyonu denedim; hiçbiri validation'da eşit ağırlığı geçemedi, olduğu gibi bıraktım.

Bir katman yoksa kalan katmanların ağırlığı kendi toplamına bölünüyor ve `available_weight` bunu raporluyor. Katman katkılarının toplamı bütün satırlarda final skora eşit; bu bir invariant ve test ediliyor. Kaydedilen model 128 satırın skorunu yeniden yüklendiğinde aynen üretiyor.

Skor bir fraud olasılığı değil, sıralama bilgisi. 0,85 "yüzde 85 fraud" demek değil.

## Adım 6: Context Adjust

Bu adım sıralamayı iyileştirmedi; sonuç aşağıda olduğu gibi duruyor.

Altı context kuralı yazdım: ürün tipine göre olağan tutar, sık ve alışıldık entity faaliyeti, alışıldık yüksek tutar, iş saati, hafta sonu programı, dış güven kaydı. Her biri tek bir katmanın katkısını azaltıyor, skorun tamamını değil; böylece hangi sinyalin bağlamla yeniden yorumlandığı izlenebiliyor. Toplam indirim 0,05 puanla ve raw skorun %15'iyle sınırlı. Güçlü entity sapması, yüksek multivariate rank, saatte 10+ işlem veya bilinen yeni bir ilişki varsa indirim tamamen bloke oluyor.

Sabit eşikte yanlış alarm 2.035'ten 1.920'ye indi. Ama yakalanan fraud da 172'den 155'e düştü. Aynı 2.207 işlemlik inceleme bütçesini iki yönteme de verdiğimde raw 172, context 170 yakaladı; sıralama iyileşmedi, sadece daha az alarm üretildi.

Ablation tablosu daha da net: ölçülen FP azalmasının tamamı tek bir kuraldan geliyor, ürün tipi tutar aralığından. İş saati, hafta sonu ve güven kuralları gerçek veride **hiç tetiklenmiyor**, çünkü IEEE-CIS takvim başlangıcı ve güven kaydı içermiyor. O üç yolu 48 sentetik API senaryosuyla test ettim; gerçek veri üzerindeki etkilerini ölçemedim.

Güven kaydı API'ye `external_context` ile giriyor ve işlemden **önce** gözlenmiş olması şartı var. İşlemle aynı saniyede gelen kayıt indirim sağlamıyor; yoksa geleceği geçmişe sızdırmış olurdum.

20 ek testle bu adımın sonunda toplam 64 test geçiyordu. Ayrıntılar [context raporunda](../reports/context_adjustment.md).

## Adım 7: Rule engine

`config/rules.json` 10 kural taşıyor: velocity kontrolleri, tutar sıçraması, yeni cihaz/ürün/e-posta ilişkileri, nadir kombinasyonlar, yüksek skor ve alışıldık faaliyet. Ülke ve gerçek gece bilgisi veride olmadığı için o tür kuralları uydurmadım.

`rules.py` JSON'daki koşul ağacını yorumluyor; `eval` veya çalıştırılabilir kod yok. Kullanılabilecek alanlar feature sözleşmesi ve belirli skor alanlarıyla sınırlı; `isFraud` ve `TransactionID` bu listede yok, yani bir kural etiketi okuyamaz.

Çakışmayı yüksek priority, eşitlikte alfabetik ID çözüyor. Alfabetik sırayı seçmemin nedeni JSON satırlarının yeri değişince kararın değişmemesi. Eşleşen ama kaybeden kurallar açıklamada `superseded` olarak duruyor.

On kuralı 590.540 işlemde çalıştırdım. Sonraki validation bölümünde 265 işlem birden fazla kurala uydu, 64'ünde review/monitor çatışması çıktı ve öncelikle çözüldü. `scripts/verify_rules.py` kayıtlı kararları pandas yerine skaler Python operatörleriyle bağımsız olarak yeniden üretiyor; örnekleme her kuralın gerçek bir tetiklenmesini ve bir çatışma örneğini zorla dahil ediyor.

Kurallar skoru değiştirmiyor, sadece yönlendiriyor. `no_rule_match` "bu işlem güvenli" demek değil.

44 ek testle proje toplamı bu adımda 108'e çıktı. [Rule engine raporu](../reports/rule_engine.md).

## Adım 8: RAG

Bilgi tabanı 17 parça: 7 politika notu ve `config/rules.json`'dan türetilen 10 kural açıklaması. Kural belgelerini elle kopyalamadım, canlı JSON'dan üretiliyorlar, yoksa kural değişince doküman sessizce yanlışlanırdı.

EmbeddingGemma 300M ile 768 boyutlu embedding üretiyorum, arama unit-normalize vektörler üzerinde exact cosine. 17 belge için approximate index veya ayrı bir vector database kurmak gereksizdi.

Generation tarafında bir şeyi bilerek kısıtladım: Qwen3 4B serbest açıklama **yazmıyor**, yalnızca getirilen kaynaklardan en fazla üçünün ID'sini seçiyor. Kararı, kazanan kuralı, gözlenen değerleri ve eşikleri `explanations.py` motor çıktısından yazıyor; seçilen kaynaklar tam metin alıntılanıyor. JSON şemasının `source_ids` alanı getirilen ID'lerle sınırlandığı için model var olmayan bir kaynağı gösteremiyor.

Bunu yapma nedenim şu: serbest metinde citation kontrolü tek başına yetmiyor. Kaynak ID'sinin listede olması, yazılan metnin motorun kararıyla tutarlı olduğunu göstermiyor; doğrulayıcının bakabileceği tek şey ID. Sahte bir model yanıtıyla test ettim: motor `review` derken "karar monitor, yüzde 85 fraud olasılığı" diyen bir yanıt, geçerli citation taşısa bile reddediliyor (`tests/test_rag.py`). Gerçek modelin böyle bir cümle kurduğunu ölçmedim; açık doğrulayıcıdaydı. Şimdi uydurma karar, sayı veya metin reddediliyor ve iki denemede düzelmezse API 502 dönüyor.

Bedeli de var: akıcılıktan feragat ettim, çıktı serbest metne göre daha kuru.

Elle yazdığım 11 geliştirme sorusunda beklenen kaynak Hit@4 içinde çıktı. 17 belgeden 4 kaynak getiriliyor, yani bu sayı genelleme başarısı ölçmez. Kaynak dışı bir soruda abstention'ı da çalıştırdım. [Gerçek yanıtlar RAG raporunda](../reports/local_rag.md).

## Adım 9: Agentic yapı

`agents.py` dört uzman ajan ve bir koordinatör içeriyor: scoring, rules, knowledge, reasoning. Görevler tipli `Task` mesajlarıyla devrediliyor; her mesaj `request_id`, `sender`, `recipient`, görev türü ve payload taşıyor. Akış scoring -> rules -> retrieval -> explanation şeklinde ilerliyor ve trace aynı `request_id` ile izleniyor.

İlk üç uzman deterministik araçlar kullanıyor, yalnız sonuncusu LLM. Bu koordinatörün yönettiği sabit bir akış; otonom planlama döngüsü veya ayrı agent process'leri yok, öyle bir iddiam da yok.

## Adım 10: API

`api.py` PDF'in istediği dört endpoint'i veriyor: `/score`, `/explain`, `/rules/evaluate`, `/rag/query`. Ek olarak `/health` servis ve dosya durumunu bildiriyor.

İki mod var: kayıtlı bir `transaction_id` veya ham `transaction` + açık `history`. İkincisinde her istek kendi `FeatureBuilder`'ını kuruyor, yani bir isteğin verisi başka bir isteğin geçmişine karışmıyor.

Hata sözleşmesi: geçersiz istek veya gelecekteki geçmiş 422, bulunamayan ID 404, eksik artifact veya erişilemeyen Ollama 503, doğrulanamayan LLM açıklaması 502.

Testler fixture bir modelle çalışıyor; ayrıca `scripts/verify_http.py` gerçek bir Uvicorn süreci açıp dört endpoint'i loopback HTTP üzerinden çağırıyor.

Bu yerel bir prototip. Kimlik doğrulama, kalıcı canlı işlem geçmişi ve yük testi yok; üretime hazır demiyorum.

## Bonus: tasarım desenleri ve DI

`container.py` `dependency_injector` ile composition root. Repository, model ve HTTP istemcisi `ThreadSafeSingleton`; koordinatör ve ajanlar `Factory`. Testler aynı provider'ları override ediyor, üretim mantığına dokunmadan.

Desenler: `FeatureRepository` (Repository), `OllamaClient` (Adapter), `ProductRiskContextEngine` (`ContextEngine`'i genişleten alternatif politika), `Task` mesajları.

## Final test

118.108 satırlık son bölümü, model/kural/context ayarları sabitlendikten sonra **bir kez** değerlendirdim. `scripts/evaluate_final.py` kaynak kod ve yapılandırma hash'lerini test etiketlerini açmadan önce yazıyor ve sonuç dosyası varsa tekrar çalışmayı reddediyor.

Aynı 4.681 inceleme kapasitesinde raw 487, context 475 fraud yakaladı. İş kuralları 8.966 incelemede 581 fraud ve 8.385 yanlış alarm üretti; farklı bütçeler, o yüzden bu sayılardan sıralama üstünlüğü çıkmaz.

Bu sonuçlardan sonra servisin varsayılan model, eşik ve kuralları değiştirilmedi. [Final rapor](../reports/final_evaluation.md).

Sonradan case'in önerdiği işlem tipi risk profilini ayrı bir profil olarak ekledim ([ürün context deneyi](../reports/product_context.md), [servis kontrolü](../reports/product_profile_verification.md)). Train etiketlerinden öğrenildiği için bu katman gözetimli. %5 bütçede 230 -> 242, %10 bütçede 457 -> 551 TP; ama %1 bütçede 46 -> 37'ye düşüyor. %5'teki 12 işlemlik farkın eşlenmiş blok bootstrap aralığı sıfırı içeriyor, %10'daki fark içermiyor ([belirsizlik ölçümü](../reports/product_uncertainty.md)). Yani kazanım kapasiteye bağlı ve %5'te kanıtlanmış değil; varsayılan yapmadım. Varsayılan `raw` kalıyor, `product_risk` karşılaştırma için `FRAUD_CONTEXT_PROFILE` ile seçiliyor. Ölçümler daha önce görülmüş validation'dan; yeni bir bağımsız test değil.

## Ek doğrulama

[Kronolojik model karşılaştırması](../reports/temporal_validation.md) zorunlu on adıma ek bir analizdir. Girdi eşleme, zaman ablation'ı, etiket gecikmesi varsayımı ve sabit inceleme bütçesiyle mevcut anomali skorunun gözetimli referans modellere göre durumunu ölçer. Bu deneyler API'nin veya anomali motorunun yerine geçmez; final test yeniden kullanılmaz. Context ve kural etkisinin değerlendirmesi 6. ve 7. adımlarda kalır.
