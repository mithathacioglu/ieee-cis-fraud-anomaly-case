# İşlem senaryoları ve bütçe karşılaştırması

Aç: `reports/scenario_demo.html`. Yeniden üret: `python scripts/scenario_demo.py`.

48 sentetik istek, gerçek kayıtlı model ve baseline context ile `/explain` üzerinden hesaplandı. Tarayıcı bu kayıtlı sonuçlar arasında geçiş yapar; canlı çıkarım yapmaz. Girdi ve çıktılar `artifacts/demo/scenarios/requests_and_responses.json` içinde yereldir.

## İki dakikalık anlatım

1. Güven indirimi: işlem ve geçmiş aynıyken, işlemden önce alınan güven kaydının sınırlı skor etkisini göster.
2. Yoğunluk koruması: aynı güven kaydıyla son saatte 12 geçmiş işleme geç; R02 inceleme önerir ve indirim bloke olur.
3. Geç gelen güven: işlemle aynı anda alınan kaydın indirim sağlayamadığını göster.
4. Hafta sonu: açık takvim tek başına yetmez; önceden bilinen faaliyet programı eklenince uygun senaryoda indirim uygulanır.
5. Bütçe: gerçek geliştirme sonuçlarında yüzde 5 kazanımı ile yüzde 1 kaybını karşılaştır; aynı inceleme sayısını koru.
6. Yerel LLM: kaydedilmiş gerçek üç işlemde dört ajan izini ve Qwen3 kaynak seçimini aç.

Takvim, güven ve işlem geçmişi senaryoları sentetiktir; gerçek IEEE-CIS müşteri bilgisi değildir. Bütçe paneli sentetik senaryolardan üretilmez: mevcut 59.054 satırlık sonraki validation sonuçlarını okur. Bu veri daha önce görüldüğü için bağımsız test değildir. Kontroller LLM çağırmaz; LLM paneli önceki gerçek yerel çağrıların kaydıdır.

Model, ağırlıklar, eşikler ve kurallar bu gösterim için değiştirilmedi. 48 senaryoda bağlamın raw katmanları değiştirmediği; güven zaman sınırı, yoğunluk koruması ve hafta sonu programı şartı doğrulandı.
