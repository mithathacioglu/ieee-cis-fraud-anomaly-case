# IEEE-CIS fraud ve anomali analizi

LOGO Data Science Case Study. Python, yerel model, ücretli API yok.

## Ne yaptım, ne çıktı

Case'in on adımını da yazdım: veri ve feature katmanı, dört anomali skoru, bağlam düzeltmesi, 10 JSON kuralı, yerel RAG, ajan görev akışı ve dört FastAPI endpoint'i. Adım adım karşılıkları ve sınırları [gereksinimlerde](docs/requirements.md). Ham veri, model dosyaları ve kimlik bilgileri pakette yok.

Sonucu baştan söyleyeyim: **context düzeltmesi işe yaramadı.** Final testte aynı 4.681 inceleme bütçesinde raw skor 487, context skoru 475 fraud yakaladı. Yani sıralamayı iyileştirmedi, sadece daha az alarm üretti. Sayıları [final raporda](reports/final_evaluation.md) olduğu gibi bıraktım.

Ölçütü de tam kurmamışım. Case'in 6. adımı false positive azaltımı istiyor, ben de onu optimize ettim: sabit eşikte FP azalsın, recall en fazla 1 puan düşsün. Hedef doğruydu, kısıt eksikti. Tolerans mutlak puan üzerinden tanımlı olduğu için, recall zaten %8 civarındayken yakalanan fraud'un onda birini kaybetmeye izin veriyor; bir de aynı inceleme kapasitesinde karşılaştırmayı seçimden *sonra* yaptım. Birincil ölçütü aynı kapasitede yakalanan fraud yapıp [aynı veride yeniden seçim koştum](reports/context_criterion.md): seçim 0 çıkıyor, yani context hiç devreye girmezdi. Yeni seçime ayrı bir göreli kayıp sınırı koymadım, çünkü kapasite sabitken TP farkı net etkiyi zaten ölçüyor; göreli kaybı aday başına raporluyorum ama eleme filtresi olarak kullanmıyorum. Sıfırdan farklı her güç eşit kapasitede daha az fraud yakalıyor. Ağırlıkları düzeltmeyi de denedim, dokuz kombinasyon ve üç politika adayı; hiçbiri fayda göstermedi.

[Ürün riski profili](reports/product_context.md) ilk bakışta işe yarıyor gibiydi: aynı 2.953 incelemeyle 230 yerine 242 fraud. Fark 12 işlem olduğu için "kazandı" demeden önce [belirsizliğini ölçtüm](reports/product_uncertainty.md). %5 kapasitede eşlenmiş blok bootstrap aralığı [-6,9, +47,1], yani sıfırı içeriyor: bu farka dayanarak politika değiştirmem. %10 kapasitede kazanım (+94) aralığı sıfırı dışlıyor ve iki dönemde de aynı yönde; %1 kapasitede işaret zarar yönünde. Yani cevap kapasiteye bağlı, "model iyi/kötü" diye tek cevabı yok. Varsayılan `raw` kalıyor, `product_risk` karşılaştırma için açıkça seçiliyor. Üstelik train etiketlerinden öğreniliyor, yani bu katman gözetimli ve ölçümler daha önce görülmüş validation'dan.

## Kurulum

Python 3.11+. Windows ve Python 3.14.6 üzerinde çalıştırdım.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[data,dev]"
```

Testlerin geçtiği paket sürümlerinin birebir aynısını istiyorsanız:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

## Veri

[IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection/data) hesabında yarışma kurallarını kabul edin. Ardından:

```powershell
.\.venv\Scripts\kaggle.exe auth login
.\.venv\Scripts\python.exe scripts/download_data.py
```

Yalnızca `train_transaction.csv` ve `train_identity.csv` indirilir. Betik ZIP bütünlüğünü açma sırasında kontrol eder; CSV kolon sayısını, benzersiz işlem kimliklerini, etiket değerlerini ve identity referanslarını doğrular. Yerel SHA-256 özetleri `data/raw/manifest.json` dosyasına yazılır. Bunlar yeniden çalıştırınca hangi dosya sürümüyle çalıştığınızı görmek için; veri sağlayıcının resmî imzası değil.

Tarayıcıdan indirdiğiniz iki CSV'yi `data/raw/` altına koyduysanız:

```powershell
.\.venv\Scripts\python.exe scripts/download_data.py --verify-only
```

Ham CSV'ler, ZIP dosyaları, kimlik bilgileri ve satır bazlı türevler Git kapsamı dışındadır.

## Veri hazırlama ve profil analizi

Proje kökünden çalıştırın:

```powershell
.\.venv\Scripts\python.exe -m fraud_case.profile
.\.venv\Scripts\python.exe -m pytest -q
```

Sonuçlar:

- [Veri profili](reports/data_profile.md): gerçek veriden üretilen bulgular ve grafik.
- `artifacts/profile/`: kolon profilleri, kategori frekansları, veri kalitesi ve ilişki özetleri.
- `data/processed/transactions.parquet`: left join sonucu, `has_identity` ve zaman bazlı split.
- `data/processed/entity_profile.parquet`: train döneminin betimleyici entity özetleri.

Sonraki %20 validation ve son %20 test için ayrılır; keşif analizi ilk %60 üzerinde çalışır. Eşit zamanlar aynı bölümde kalır. Bölümlerin gerçek boyutları rapora yazılır.

## Tasarım notları

- `TransactionDT` takvim tarihi değil, açıklanmamış bir referanstan geçen saniye. Başlangıç tarihi uydurmadım.
- `isFraud` hedef, `TransactionID` anahtar. İkisini de modele sokmuyorum.
- Identity kaydı yok diye işlem silmiyorum; join `one_to_one` ile doğrulanıyor.
- Kart/adres kombinasyonu gerçek müşteri kimliği değil. Bileşeni eksik olanları tek bir "bilinmeyen" grubunda toplamadım; öyle yapsam alakasız satırlar aynı geçmişi paylaşırdı.
- Veri katmanında null doldurmuyorum, outlier silmiyorum. Doldurma kararı model katmanına ait ve orada da yalnız train'de öğrenilir.

Kararların gerekçesi [teknik notlarda](docs/technical.md).

## Feature engineering

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m fraud_case.build_features
.\.venv\Scripts\python.exe scripts/verify_features.py
```

Bu komut `data/processed/features.parquet`, `artifacts/features/contract.json`, `artifacts/features/summary.json` ve [feature raporunu](reports/features.md) üretir. 590.540 işlemde çalıştırılmıştır. Bağımsız doğrulama betiği geçmiş sayıları, ortalamalar ve velocity pencerelerini farklı bir batch hesabıyla kontrol eder.

`FeatureBuilder` aynı saniyedeki bütün satırları önce hesaplıyor, sonra geçmişe ekliyor; kaynak saniye veriyor, sıra vermiyor. Etiketi hiç okumuyor; `build_features.py` parquet'i zaten sadece gerekli kolonlarla açıyor. Validation ve test boyunca geçmiş kronolojik olarak güncelleniyor ama yalnız etiketsiz gözlemlerle.

Takvim referansı vermezseniz saat ve hafta sonu alanları null kalıyor. `--calendar-reference` ile UTC offset'li bir ISO tarih verebilirsiniz, ama o tarih sizin varsayımınız; IEEE-CIS böyle bir şey yayımlamıyor.

## Anomali katmanları ve skor birleştirme

```powershell
.\.venv\Scripts\python.exe -m fraud_case.build_scores
```

Yöntem ve parametreler [skorlama raporunda](reports/anomaly_scores.md) açıklanır. `config/scoring.json` minimum entity geçmişini, IsolationForest ayarlarını ve dört katmanın ağırlıklarını içerir. Eğitim ve normalizasyon referansları yalnızca train üzerinde fit edilir. Bu komut da etiketleri okumaz.

- `data/processed/scores.parquet`: katman raw/normalized skorları, availability, etkin ağırlıklar, katkılar ve `raw_anomaly_score`.
- `artifacts/scoring/model.joblib`: yerelde üretilmiş motor ve normalizasyon referansları.
- `artifacts/scoring/summary.json`: yapılandırma, kaynak parmak izi ve kapsama.
- `artifacts/scoring/explanation.json`: bir validation işleminin sayısal açıklaması.

Skor bir fraud olasılığı değil, sıralama bilgisi. Geçmişi yetersiz işlemde entity ve temporal katmanlar null dönüyor; sıfır değil, çünkü geçmişi olmayan bir işlem bilinmiyordur, güvenli değil. Kalan katmanların ağırlığı kendi toplamına bölünüyor.

IsolationForest için feature attribution hesaplamadım. Gözlenen girdileri ve alt bileşenleri raporluyorum; uydurma bir SHAP açıklaması üretmiyorum.

## Context adjustment ve değerlendirme

```powershell
.\.venv\Scripts\python.exe -m fraud_case.evaluate_context
```

Kurallar `config/context.json`, deney koşulları `config/context_evaluation.json` içinde. Ürün grubu tutar referanslarını train'de öğreniyorum, indirim gücünü erken validation'da seçiyorum, sonraki yarıda aynı eşik ve aynı bütçeyle kontrol ediyorum. O ikinci yarı bağımsız bir test değil; geliştirme sırasında zaten görüldü.

[Context raporu](reports/context_adjustment.md) kural gerekçelerini, yanlış alarm azalmasının recall bedelini, dış bağlam eksiklerini ve deney sınırlarını içerir. Sayısal çıktılar `artifacts/context/evaluation.json`, işlem bazlı açıklanabilir skorlar `data/processed/context_scores.parquet` dosyasındadır.

Uygulamada `ContextEngine` seçilmiş yapılandırma ve train ürün referanslarıyla kurulur:

```python
import json
from pathlib import Path
from fraud_case.context import ContextConfig, ContextEngine

folder = Path("artifacts/context")
config = ContextConfig(**json.loads((folder / "selected_config.json").read_text()))
references = json.loads((folder / "product_reference.json").read_text())
context = ContextEngine(config, references)
# features ve scores aynı TransactionID/index sırasına sahip olmalıdır.
# adjusted = context.apply(features, scores, external=external_context)
```

Dış güven kaydı veya hafta sonu programı, `TransactionID` ile eşlenmiş bir DataFrame olarak verilebiliyor. Bayrağın yanında kaynak ve gözlem zamanı istiyorum, zaman da işlemden **önce** olmak zorunda. İşlemle aynı saniyede gelen kayıt indirim sağlamıyor; yoksa geleceği geçmişe sızdırmış olurdum. Sık işlem yapan entity'yi kendiliğinden güvenilir saymıyorum; saldırgan da sık işlem yapabilir.

Aynı bağlam API'ye `external_context` ile giriyor; alanlar ve örnek [API sözleşmesinde](docs/runtime.md). Altı senaryonun kayıtlı modelle sonucu [bağlam gösteriminde](reports/context_api.md). Senaryolar sentetik; IEEE-CIS ne takvim ne güven kaydı veriyor.

## JSON rule engine

```powershell
.\.venv\Scripts\python.exe -m fraud_case.evaluate_rules
.\.venv\Scripts\python.exe scripts/verify_rules.py
```

[config/rules.json](config/rules.json) 10 kural taşıyor: velocity, tutar sıçraması, yeni ilişkiler, nadir kombinasyonlar, yüksek skor ve alışıldık faaliyet. Ülke ve gerçek gece bilgisi veride olmadığı için o tür kuralları uydurmadım.

`when` koşulları `all` / `any` ile birleşiyor, `then.action` review veya monitor veriyor. Çakışmayı yüksek priority çözüyor; eşitlikte alfabetik ID. Alfabetiği seçmemin nedeni JSON satırlarının yeri değişince kararın değişmemesi. Hiçbir kural eşleşmezse `no_rule_match` dönüyor; bu "işlem temiz" demek değil. Kurallar skoru değiştirmiyor, sadece yönlendiriyor.

```python
from fraud_case.rules import RuleEngine

engine = RuleEngine.from_json("config/rules.json")
# inputs: TransactionID ve kuralların kullandığı feature/skor kolonları.
# decisions = engine.evaluate(inputs)
# explanation = engine.explain(inputs.iloc[[0]])
```

JSON'u değiştirip yeni bir motor örneği oluşturmak yeterli; Python'a dokunmaya gerek yok. Özel dosyayı `--rules config/my_rules.json` ile veriyorsunuz. Her açıklamada gözlenen koşullar, eşleşen/kazanan/elenen kurallar ve yapılandırma hash'i duruyor. Bilinmeyen alan, geçersiz operatör veya eksik kolon hata veriyor; ve hiçbir yerde kod çalıştırılmıyor, koşullar doğrulanmış literal.

[Rule engine raporu](reports/rule_engine.md) gerçek eşleşme ve çakışma örneklerini, öncelik politikasını ve inceleme etkisini içerir. Geliştirme validation bölümünde context eşiği 0.85 ile 2.135 incelemede 162 fraud bulunurken kurallar 3.667 incelemede 236 fraud buldu; FP 1.973'ten 3.431'e çıktı. Kurallar 74 fraud'u fazladan yakalıyor ama bunun için 1.532 ek inceleme istiyor. İki sayı aynı bütçeden gelmediği için doğrudan karşılaştırılamaz.

Çıktılar `data/processed/rule_decisions.parquet` ve `artifacts/rules/` altındadır. Son kullanılan JSON'un kopyası, girdi hash'leri, ölçümler ve ayrıntılı açıklama örnekleri saklanır. Bağımsız doğrulama betiği kayıtlı kararların bir örneklemini ayrı bir satır bazlı hesapla yeniden üretir; etiket okumaz.

## Yerel LLM ve RAG

Windows'ta bir defalık kurulum:

```powershell
.\.venv\Scripts\python.exe scripts/setup_ollama.py
.\.venv\Scripts\python.exe -m fraud_case.build_rag
```

Betik resmî Ollama Windows paketini sürüm ve SHA-256 kontrolüyle `.tools/ollama` içine kurar. EmbeddingGemma 300M ve Qwen3 4B modelleri `models/ollama` altında tutulur; indirme toplamı yaklaşık 4.6 GB, açılmış dosyaların disk ihtiyacı daha fazladır. Servis penceresiz, `127.0.0.1:11435` üzerinde ve `OLLAMA_NO_CLOUD=1` ile başlar. Sistem PATH'ini veya başlangıç uygulamalarını değiştirmez. Daha sonra internet gerekmeden başlatmak için `scripts/setup_ollama.py --start-only` kullanılır.

Linux/macOS'ta Ollama ayrıca kurulmalıdır. Aynı modelleri indirip servisi `OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NO_CLOUD=1 ollama serve` ile çalıştırın; uygulamanın `config/rag.json` adresiyle eşleşmesi gerekir. Bu platformlar burada çalıştırılmadı.

Bilgi tabanı 7 açıklama/politika notu ve canlı JSON'dan türetilen 10 kural parçasıdır. 768 boyutlu embedding ve exact cosine search kullanılır. On yedi belge için ayrı bir vector database servisi kurmak gereksiz; tam arama hem basit hem denetlenebilir. Kaynaklar modele soruyla birlikte gönderilir; yanıtın JSON şeması ve kaynak ID'leri doğrulanır. Bilgi tabanı/kural dosyası veya embedding modeli değişirse index yeniden üretilmelidir.

Burada bir şeyi bilerek kısıtladım: model **açıklama yazmıyor**, sadece ilgili kaynakların ID'sini seçiyor. Kararı, kazanan kuralı, gözlemleri ve eşikleri motor çıktısından ben yazıyorum; seçilen kaynaklar tam metin alıntılanıyor.

Nedeni şu: serbest metinde citation kontrolü tek başına yetmiyor. Kaynak ID'sinin listede olması, yazılan metnin motorun kararıyla tutarlı olduğunu göstermiyor; doğrulayıcının bakabileceği tek şey ID. Bunu sahte bir model yanıtıyla test ettim: motor `review` derken "karar monitor, yüzde 85 fraud olasılığı" diyen bir yanıt, geçerli citation taşısa bile reddediliyor (`tests/test_rag.py`). Gerçek modelin böyle bir cümle kurduğunu ölçmedim; açık doğrulayıcıdaydı, onu kapattım.

Şimdi uydurma karar, sayı veya metin reddediliyor; iki denemede düzelmezse 502 dönüyor. Yerel modele ulaşılamazsa 503. Eşiği geçen kaynak çıkmazsa yanıt abstain ediyor ama işlem kanıtı yine gösteriliyor.

Bedeli de var: akıcılıktan feragat ettim, çıktı serbest metne göre kuru. Kaynak seçiminin soruyla ilgisi ve bilgi tabanının doğruluğu hâlâ ayrı bir değerlendirme konusu. [Örnek yanıtlar](reports/local_rag.md), [servis mimarisi](docs/runtime.md).

## Üç işlemle gösterim

**Kurulumsuz inceleme:** [İşlem laboratuvarını](reports/scenario_demo.html) indirip tarayıcıda açın. 48 hesaplanmış sentetik senaryoda tutar, yoğunluk, güven zamanı ve takvim değiştirilebilir; ölçülen geliştirme sonuçları %1/%5/%10 inceleme bütçesinde karşılaştırılabilir. Aynı dosyada daha önce yerel Qwen3 ile çalıştırılmış üç işlemin ajan izi ve kaynaklı açıklaması var. Tarayıcı canlı model çalıştırmaz; kayıtlı sonuçlar arasında geçiş yapar. [İki dakikalık anlatım rehberi](reports/scenario_demo.md).

Senaryo sonuçlarını mevcut modelle yeniden hesaplamak için, aşağıdaki üç işlem demosundan sonra `python scripts/scenario_demo.py` çalıştırılır. Model, eşik ve kurallar gösterim için değiştirilmez.

Hazırlık tamamlandıktan sonra:

```powershell
.\.venv\Scripts\python.exe scripts/setup_ollama.py --start-only
.\.venv\Scripts\python.exe scripts/demo.py
```

`artifacts/demo/full/index.html` dosyasını tarayıcıda açın. Dış script, font veya CDN gerektirmez; kayıtlı gerçek API çıktısını gösterir. Komut tekrar çalıştırıldığında üç işlem yeniden değerlendirilir ve yerel model yeniden kaynak seçer. Alışıldık faaliyet, yüksek skorlu işlem ve review/monitor çatışması; her birinde katmanlar, eşikler, kural önceliği, ajan izi ve kaynaklar görünür. Üç kayıt da fraud etiketlerine bakılmadan seçildi. Amaç mekanizmayı göstermek, başarı ölçmek değil.

Hızlı, LLM'siz gösterim: `python scripts/demo.py --no-rag`. Bu sürüm ayrı `artifacts/demo/no_rag/` klasörüne yazılır ve LLM kullanılmadığını açıkça belirtir. [Gösterim rehberi](reports/demo.md) anlatım sırasını içerir.

## API

Varsayılan profil `raw`: context indirimi uygulanmaz, `adjusted_anomaly_score` ham skora eşit çıkar. Context deneyi sabit bütçede kazandırmadığı için varsayılanı böyle bıraktım. Davranış context'ini kayıtlı sonuçlarla birebir çalıştırmak için `behavior_context`, case'in işlem tipi risk bağlamını karşılaştırmak için `product_risk` seçilir. Temel hazırlık komutlarından sonra:

```powershell
.\.venv\Scripts\python.exe scripts/evaluate_product_context.py
.\.venv\Scripts\python.exe scripts/verify_product_profile.py
$env:FRAUD_CONTEXT_PROFILE = "product_risk"
.\.venv\Scripts\python.exe -m uvicorn fraud_case.api:app --host 127.0.0.1 --port 8000
```

Servisi durdurup `$env:FRAUD_CONTEXT_PROFILE = "raw"` ile varsayılana dönülür. Her skor yanıtı ve `/health` aktif profili bildirir. Yeni profilin gerçek modeli kullanan API ve [kural etkisi doğrulaması](reports/product_profile_verification.md) ayrı raporlanır. Kural eşikleri değiştirilmediğinden ek inceleme yükü vardır; skorun sabit bütçeli kazanımı kural motorunun aynı bütçede çalıştığı anlamına gelmez.

Hazırlık komutları tamamlandıktan sonra proje kökünde:

```powershell
.\.venv\Scripts\python.exe -m uvicorn fraud_case.api:app --host 127.0.0.1 --port 8000
```

| Endpoint | Görev | Örnek JSON |
|---|---|---|
| POST `/score` | Gerçek modelle skor + kural kararı | `{"transaction_id":3400481}` |
| POST `/explain` | Katman/context/kural açıklaması ve yerel RAG | `{"transaction_id":3400481,"include_rag":true}` |
| POST `/rules/evaluate` | Eşleşmeler, kazanan ve çakışma açıklaması | `{"transaction_id":3400481}` |
| POST `/rag/query` | Politika bilgisine dayalı yanıt | `{"question":"Sık işlem yapan entity güvenilir midir?"}` |

```python
import httpx

response = httpx.post("http://127.0.0.1:8000/score", json={"transaction_id": 3400481})
response.raise_for_status()
print(response.json())
```

Kayıtlı ID yerine `transaction` ve isteğe bağlı `history` gönderilebilir; [API örnekleri](docs/runtime.md) alanları ve sınırları açıklar. `/explain` için `include_rag=false` seçilirse Ollama gerekmez. `/health` servis ve dosya durumunu, `/openapi.json` şemayı verir. `/docs` Swagger arayüzüdür; varsayılan JS/CSS varlıkları CDN kullanır, API'nin kendisi offline çalışır.

Bu yerel bir prototip: kimlik doğrulama, kalıcı işlem kuyruğu, rate limiting ve dağıtık history store yok. Üretime hazır demiyorum. Geçmiş feature'lar ya read-only snapshot'tan ya isteğin kendi geçmişinden geliyor; her raw-history isteği kendi `FeatureBuilder`'ını kuruyor, yani bir isteğin verisi başka bir isteğe karışmıyor.

## Ek doğrulama: model karşılaştırması

Bu bölüm case'in zorunlu adımlarına ek bir değerlendirmedir. Servisteki dört anomali katmanı, context, kurallar ve RAG akışı aynı kalır.

İlk [gözetimli karşılaştırmada](reports/supervised_baseline.md), aynı 2.953 incelemede ham anomali skoru 230, gradient boosting 628 fraud yakaladı. Girdiler eşlenmiş değildi: gözetimli modeller 36, anomali katmanları toplam 26 feature kullanıyordu. Fark yalnız etiket kullanımına atfedilemez. Eşlenmiş satır bootstrap'ının %95 aralığı [348, 444]; bu hesap zaman ve entity bağımlılığını korumadığı için gelecek dönemlere ilişkin bir güvence vermez.

[Kronolojik değerlendirmede](reports/temporal_validation.md) üç dönem, 0 ve 7 günlük etiket gecikmesi varsayımları ve üç girdi kümesi karşılaştırıldı. Gecikme, veri setindeki işlemin etiketinin ne zaman hazır sayıldığıdır; projenin hazırlanma süresi değildir. Eşlenmiş 26 feature ile gradient boosting, her iki gecikme senaryosunun üç döneminde de ham anomali skorundan daha fazla fraud yakaladı. Denenen %50 kotalı hibrit, aynı bütçede eşlenmiş gradient boosting'in gerisinde kaldı. Bu sonuç hibrit kullanımı desteklemedi.

Deney ayrıntıları [protokolde](docs/validation_protocol.md). Bunlar daha önce incelenmiş train/validation üzerinde geliştirme sonuçlarıdır; final test yeniden kullanılmadı. Sayısal özet [JSON olarak](reports/temporal_validation_summary.json) da bulunur. Kaydedilen tahminlerden yeniden hesaplanan 1.114 kontrol geçti ([kontrol sonucu](reports/temporal_validation_checks.json)).

```powershell
python scripts/temporal_validation.py --data-root data/processed --output artifacts/temporal_validation
python scripts/report_temporal_validation.py
python scripts/verify_temporal_validation.py
```

## Doğrulama ve teslim

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts/verify_context.py
.\.venv\Scripts\python.exe scripts/verify_context_api.py
.\.venv\Scripts\python.exe scripts/verify_rules.py
.\.venv\Scripts\python.exe scripts/verify_platform.py
.\.venv\Scripts\python.exe scripts/verify_http.py
.\.venv\Scripts\python.exe scripts/verify_http.py --context-profile product_risk
.\.venv\Scripts\python.exe scripts/verify_demo_html.py
```

Değerlendirme tarafındaki ek deneyler ayrı çalışır ve dondurulmuş çıktıların hiçbirine yazmaz:

```powershell
.\.venv\Scripts\python.exe scripts/reselect_context_strength.py
.\.venv\Scripts\python.exe scripts/evaluate_product_uncertainty.py
.\.venv\Scripts\python.exe scripts/evaluate_default_policy.py
```

Sırasıyla [seçim ölçütünü](reports/context_criterion.md), [ürün riski kazanımının belirsizliğini](reports/product_uncertainty.md) ve [varsayılan profilin kural kararlarını](reports/default_policy.md) raporlar.

Unit/integration testleri Kaggle dosyaları veya LLM indirmeden çalışır; API testleri küçük yerel fixture modeli, LLM sınır testleri açık test doubles kullanır. Son iki betik ise yerel Qwen3 ve EmbeddingGemma'yı çağırır. HTTP kontrolü geçici Uvicorn sürecini penceresiz açar ve bitince yalnızca kendi açtığı süreci kapatır. Aynı test paketi için GitHub Actions tanımı `.github/workflows/tests.yml` içinde hazır; bağımlılıkları `.[dev]` aralıklarından değil `requirements.lock` içindeki sabit sürümlerden kurar, sonra `ruff check` ve `pytest` çalıştırır. Yani CI yerelde doğruladığım kapanışın aynısını kuruyor.

Teslim hazırlığı: demo ve yerel model doğrulamaları bittikten sonra `python scripts/prepare_delivery.py`, `python scripts/package_submission.py`, `python scripts/verify_package.py`. Son komut ZIP'i yeni bir geçici dizine açar; dosya hash'lerini ve Markdown bağlantılarını kontrol eder, testleri paketin kaynaklarından çalıştırır. Paket ham veri, yerel model veya kimlik bilgisi içermez.

Test sayısı, gerçek loopback HTTP sonuçları, model digest'leri ve final metrikler [doğrulama özetindedir](reports/verification_summary.json). O dosyadaki bütün sayılar betiklerin çıktısından üretilir; elle tuttuğum ikinci bir sayı listesi bilinçli olarak yok, çünkü elle tutulan liste kaçınılmaz olarak kayıyor. Paketleme öncesinde test sonucu `python -m pytest -q --junitxml=artifacts/platform/pytest.xml` ile kaydedilir; `scripts/prepare_delivery.py` başarılı test sayısını bu dosyadan okur. Küçük yerel modelin özellikle Türkçe ifade kalitesi sınırlıdır; sayısal açıklamalar ve kaynaklar model metninden ayrı sunulur.

Final testi ayarlar sabitlendikten sonra `scripts/evaluate_final.py` ile bir kez çalıştırdım. Betik kaynak kod ve yapılandırma hash'lerini test etiketlerini açmadan önce yazıyor ve sonuç dosyası varsa tekrar çalışmayı reddediyor. Model dosyası, inceleme eşiği ve `config/rules.json` o günden beri değişmedi. Ama bu "kararlar da değişmedi" diye okunmamalı: varsayılan context profilini final testten **sonra** `raw` yaptım ve R09 eşiği `adjusted_anomaly_score` üzerinde çalıştığı için kural kararları değişiyor. Final raporun Rules satırı eski `behavior_context` politikasına aittir; farkı geliştirme verisinde ölçtüm ([varsayılan politika](reports/default_policy.md)), final dönemi için bilmiyorum. Sonraki model karşılaştırmaları ayrı geliştirme deneyleridir. [Final rapor](reports/final_evaluation.md).

Teslim paketi `scripts/package_submission.py` ile oluşturulur. Allowlist yalnızca kaynak, test, yapılandırma ve raporları alır; ham veri, yerel model ağırlıkları, `.venv`, çalışma artifact'ları ve kimlik bilgileri alınmaz. Paket içerik hash'lerini de taşır.
