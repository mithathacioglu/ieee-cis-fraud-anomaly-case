# Katmanların aynı bütçede katkısı

Dört anomali katmanının dört ayrı yararlı sinyal olup olmadığı bu teslimde ölçülmemiş bir soruydu. Tutar, geçmiş ve velocity birden fazla katmanda görünüyor, dolayısıyla eşit ağırlıklı birleşimin gerçekten dört bağımsız kanıt topladığı kendiliğinden doğru değil. Burada iki şeyi ölçüyorum: her katman tek başına aynı kapasitede ne yakalıyor, ve bir katmanı çıkarınca ne kaybediyorum.

Segment: validation'ın audit yarısı, 59,054 işlem, 2,061 fraud. Görülmüş geliştirme verisi; bağımsız holdout değil.

Eşit ağırlıklı yeniden birleştirme kayıtlı raw skoru birebir üretiyor (en büyük fark 0.0e+00), yani birini-çıkar hesabı servisin yaptığı işlemin aynısını kullanıyor. Hiçbir katmanı olmayan satır o varyantla sıralanamaz ve en sona konur; sıfır risk sayılmaz.

## Aynı kapasitede yakalanan fraud

| Sıralama | TP %1 | TP %5 | TP %10 | Sıralanamayan satır |
|---|---:|---:|---:|---:|
| Dördü birlikte | 46 | 230 | 457 | 0 |
| Yalnız column | 22 | 207 | 366 | 0 |
| column çıkarılmış | 41 | 268 | 657 | 0 |
| Yalnız multivariate | 33 | 235 | 545 | 0 |
| multivariate çıkarılmış | 25 | 201 | 391 | 0 |
| Yalnız entity | 12 | 118 | 221 | 12,622 |
| entity çıkarılmış | 49 | 232 | 441 | 0 |
| Yalnız temporal | 27 | 96 | 199 | 12,622 |
| temporal çıkarılmış | 52 | 233 | 438 | 0 |

İnceleme sayıları: %1 = 591, %5 = 2,953, %10 = 5,906.

## Birincil bütçede fark ve belirsizlik

Aşağıdaki aralıklar %5 kapasitedeki TP farkına ait, eşlenmiş blok bootstrap ile. Kanıt okuması ürün riski raporundaki fonksiyonun aynısı: aralığın hesaplanıp hesaplanmadığı, sıfırı dışlayıp dışlamadığı, yönünün gözlenen farkla uyuşup uyuşmadığı ve iki dönemde aynı yönde olup olmadığı ayrı ayrı soruluyor.

| Karşılaştırma | Gözlenen | İlk yarı | İkinci yarı | Okuma |
|---|---:|---:|---:|---|
| Yalnız column - dördü birlikte | -23 | +1 | -21 | kanıt yok |
| column çıkarılmış - dördü birlikte | +38 | +39 | -1 | kanıt yok |
| Yalnız multivariate - dördü birlikte | +5 | +22 | -18 | kanıt yok |
| multivariate çıkarılmış - dördü birlikte | -29 | -14 | -14 | yön tutarlı ama belirsizlik sıfırı kapsıyor |
| Yalnız entity - dördü birlikte | -112 | -48 | -56 | bu pencerede zarar var |
| entity çıkarılmış - dördü birlikte | +2 | -2 | -1 | yön tutarlı ama belirsizlik sıfırı kapsıyor |
| Yalnız temporal - dördü birlikte | -134 | -48 | -84 | bu pencerede zarar var |
| temporal çıkarılmış - dördü birlikte | +3 | -3 | +2 | kanıt yok |

## Okuma

Çıkarıldığında işaret kayıp yönünde olan ama aralığı sıfırı kapsayan katmanlar: multivariate. Bunları "kayıp vermiyor" diye yazmam; fark iki dönemde de aynı yönde, yalnızca sıfırdan ayrılmıyor.
Çıkarıldığında bu pencerede kayıp işareti bile görülmeyen katmanlar: column, entity, temporal. Bu, o katmanların gereksiz olduğunu kanıtlamıyor; bu kapasitede ve bu dönemde farkın sıfırdan ayrılmadığını söylüyor.
Tek başına birleşimden belirgin biçimde kötü olan katmanlar: entity, temporal. Bu beklenen sonuç: tek katman daha az bilgi görüyor.
En büyük gözlenen etki temporal tarafında (-134) ve okuması "bu pencerede zarar var". Gözlenen en büyük sayının kanıt olarak en güçlü sayı olmadığını ayrıca yazıyorum, çünkü tabloya bakan önce ona bakar.

Ağırlıkları bu sonuca göre değiştirmedim ve değiştirmeyi önermiyorum. Bu segment context geliştirmesinde zaten görüldü; buradan ağırlık seçmek aynı veriye ikinci kez uymak olur. Katman girdileri tasarım gereği örtüşüyor, dolayısıyla tek katman sonucu bir bağımsızlık testi değil. Sonraki adım, bu ölçümü görülmemiş bir dönemde tekrarlamak olurdu.

Yeniden çalıştırma: `python scripts/evaluate_layer_contribution.py`. Çıktı: `artifacts/layer_contribution/evaluation.json`.
