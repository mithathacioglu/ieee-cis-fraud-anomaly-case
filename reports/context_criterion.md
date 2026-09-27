# Seçim ölçütünü yeniden kurmak

Context deneyine başlarken başarı ölçütünü şöyle kurmuştum: sabit eşikte false positive azalsın, recall en fazla 1 yüzde puan düşsün. Teslimdeki 0.5 gücünü seçen de bu ölçüttü.

Hedef tarafı case'in istediği şeydi: azaltılacak olan false positive. Eksik olan kısıt tarafı. Tolerans mutlak puan üzerinden tanımlı; baseline recall %8 civarında olduğu için 1 puanlık izin, yakalanan fraud'un yaklaşık onda birini kaybetmek anlamına geliyor. Ölçüt bu oranı hiç görmüyor. Birincil ölçüt verilen inceleme kapasitesinde yakalanan fraud olmalıydı: kapasite sabitken daha az alarm üretmek kazanç değildir, daha çok fraud yakalamak kazançtır. O karşılaştırmayı yaptım ama seçimden sonra.

Aşağıdaki yeniden seçimde ayrı bir göreli kayıp sınırı **uygulamıyorum.** Kapasite sabitken TP farkı kaybı da kazancı da aynı sayıda ölçüyor, ikinci bir tolerans gereksiz olurdu. Göreli kaybı tabloda aday başına raporluyorum ama eleme filtresi olarak kullanmıyorum; seçim sırası kapasitedeki TP farkı, sonra AP, sonra küçük güç. Bunu ayrıca yazıyorum çünkü "kısıtı düzelttim" demek, kodda olmayan bir filtre ima eder.

Bunu geçmişe dönük düzeltmiş gibi yazmıyorum. Aşağıdaki tablo, ilk seçimin kullandığı aynı calibration yarısında (59,054 işlem) iki ölçütün ne seçtiğini yan yana koyuyor. Yeni ölçütün bağımsız doğrulaması değil; ölçülen şey ölçütün kararı nasıl değiştirdiği.

Eşik train skorlarından sabitlenmiş: 0.85167813. Eşit kapasite karşılaştırması ham skorun bu eşikte ürettiği 2,436 incelemeyi kullanıyor.

| Güç | Sabit eşikte kaldırılan FP | Kaybedilen fraud | Kaybın oranı | Recall düşüşü | Eşit kapasitede fraud farkı |
|---|---:|---:|---:|---:|---:|
| 0.0 | +0 | +0 | %0.00 | 0.000 puan | +0 |
| 0.25 | +46 | +12 | %4.35 | 0.471 puan | -8 |
| 0.5 | +97 | +23 | %8.33 | 0.902 puan | -19 |
| 1.0 | +152 | +29 | %10.51 | 1.137 puan | -22 |

## İki ölçüt ne seçiyor

- İlk ölçüt (teslimdeki): **0.5**. Teslimde çalışan güç de 0.5, yani tablo teslimi yeniden üretiyor.
- Kapasite ölçütü: **0.0**, eşit kapasitede fraud farkı +0.


Hiçbir aday eşit kapasitede kazandırmıyor; en iyisi 0.0 gücü ve onun farkı da +0. Teslimdeki 0.5 gücü -19 veriyor, yani sıfırdan farklı her güç aynı kapasitede daha az fraud yakalıyor. Bu durumda context'in devreye girmesinin sebebi verideki bir kazanım değil, ölçütün eksik kurulmasıydı. Servis varsayılanının `raw` olması bu sonuçla tutarlı; farkı artık "audit sonucunu beğenmedim" diye değil, "ölçütü eksik kurmuşum" diye söylüyorum.

## Sınırlar

Bu hesap calibration yarısını ikinci kez kullanıyor. Yeni ölçütün doğruluğunu kanıtlamaz, yalnızca ilk ölçütün bu veride nasıl bir karar ürettiğini gösterir. Dondurulmuş çıktıların hiçbirine yazmıyor: `artifacts/context/selected_config.json`, `data/processed/context_scores.parquet` ve final artifact'ları değişmedi. Final test etiketleri okunmadı.

Aday güçler (0.0, 0.25, 0.5, 1.0) ilk deneyde etiket görülmeden yazılmıştı; bu koşu aynı listeyi kullanıyor, yeni aday eklemiyor.

Yeniden çalıştırma: `python scripts/reselect_context_strength.py`. Çıktı: `artifacts/context_criterion/evaluation.json`.
