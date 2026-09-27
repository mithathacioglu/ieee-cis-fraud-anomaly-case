# İşlem tipi risk profili ile context deneyi

Case'in 6. adımında önerilen işlem tipi bazlı risk profili uygulanır. Dört anomali katmanı ve eşit ağırlıklı raw skor değişmez. Önceki ürün grubunda alışıldık tutar indirimi, geçmiş etiketlerden öğrenilen ürün riski ile değiştirilir; diğer davranış ve takvim/güven kuralları korunur.

Risk profili yalnızca train etiketlerinden hesaplanır. 1.000 sanal gözlemle genel fraud oranına yumuşatma ve en az 1.000 gerçek gözlem desteği kullanılır. Log risk oranı [-1,1] ile sınırlanır; skor düzeltmesi en fazla seçilen güç kadardır. Güçlü anomali koruması negatif düzeltmeyi bloke eder. Tanınmayan ürün düzeltme almaz. Skor yine kalibre edilmiş fraud olasılığı değildir.

Train fraud etiketlerinin validation başlamadan bilindiği varsayılır; veri fraud bildirim gecikmesi sağlamaz. Bu nedenle context katmanı gözetimlidir. Önceki final test görülmüş olduğundan sonraki validation bir geliştirme kontrolüdür, bağımsız başarı kanıtı değildir. Test etiketleri bu deneyde okunmaz.

| Ürün | Train işlem | Train fraud | Yumuşatılmış oran | Sınırlı log risk oranı |
|---|---:|---:|---:|---:|
| C | 41081 | 4412 | 10.565% | 1.0000 |
| H | 26612 | 1133 | 4.226% | 0.2223 |
| R | 28182 | 909 | 3.231% | -0.0461 |
| S | 6314 | 379 | 5.644% | 0.5118 |
| W | 252135 | 5155 | 2.050% | -0.5011 |

## Güç seçimi

Erken validation iki kronolojik blokta değerlendirilir; aynı %5 inceleme bütçesinde en düşük blok TP kazanımı, sonra toplam TP ve AP kullanılır. Dört güç adayı etiketler okunmadan planda kaydedilir.

| Güç | Erken validation TP | En düşük blok TP kazanımı | AP |
|---|---:|---:|---:|
| 0.0 | 324 | 0 | 0.0822 |
| 0.025 | 326 | 2 | 0.0853 |
| 0.05 | 336 | 3 | 0.0881 |
| 0.1 | 340 | 7 | 0.0932 |

Seçilen güç: **0.1**.

## Sonraki validation: aynı inceleme bütçesi

| Bütçe | Yöntem | İnceleme | TP | FP | Precision | Recall |
|---|---|---:|---:|---:|---:|---:|
| 1% | raw | 591 | 46 | 545 | 7.783% | 2.232% |
| 1% | original_context | 591 | 46 | 545 | 7.783% | 2.232% |
| 1% | product_context | 591 | 37 | 554 | 6.261% | 1.795% |
| 5% | raw | 2953 | 230 | 2723 | 7.789% | 11.160% |
| 5% | original_context | 2953 | 227 | 2726 | 7.687% | 11.014% |
| 5% | product_context | 2953 | 242 | 2711 | 8.195% | 11.742% |
| 10% | raw | 5906 | 457 | 5449 | 7.738% | 22.174% |
| 10% | original_context | 5906 | 458 | 5448 | 7.755% | 22.222% |
| 10% | product_context | 5906 | 551 | 5355 | 9.329% | 26.735% |

AP raw: 0.0634; ürün context: 0.0716.

## Train'den seçilen eşiklerle alarm yükü

Bu eşikler yaklaşık train üst %5'inden ayrı ayrı seçilir. Validation'da eşit alarm bütçesi garantilemez; yukarıdaki sabit bütçe kıyasıyla karıştırılmamalıdır.

Ürün riski yalnızca train dönemi bittikten sonra, yani TransactionDT > 8745772 olan işlemlerde uygulanır. Servis bu kapıyı zaten uyguluyordu, deney uygulamıyordu: eşik train satırlarına ürün düzeltmesi eklenerek hesaplanıyordu, yani servisin o satırlarda hiç üretmediği bir skor dağılımından. Deneyi servise eşitledim. Sözle bırakmamak için seçilen politikayı servisin kendi motoruna verip 4000 satırda karşılaştırdım; kesimin iki tarafını da kapsıyor ve en büyük fark 0.

Düzeltmenin görünür sonucu: train üst %5'i artık davranış skorundan hesaplandığı için ürün context'inin eşiği ham skorun eşiğiyle aynı çıkıyor. Yani aşağıdaki satırlar aynı eşikte karşılaştırma. Ürün riski skorları yukarı taşıdığı için eşiği geçen işlem sayısı artıyor; bu bir iyileşme değil, daha büyük bir inceleme yükü. Sabit bütçeli karşılaştırma yukarıdaki tabloda.

| Yöntem | Eşik | Alarm | TP | FP |
|---|---:|---:|---:|---:|
| raw | 0.851678 | 2207 | 172 | 2035 |
| product_context | 0.851678 | 3707 | 305 | 3402 |

Bu adayın raporu özgün final değerlendirmesinin yerine geçmez. Kural motoru ayrı değerlendirilmelidir; daha iyi skor sıralaması her sabit iş kuralının kararını otomatik iyileştirmez.

Yeniden çalıştırma: `python scripts/evaluate_product_context.py`. Parametre ve çıktı: `artifacts/product_context/`.
