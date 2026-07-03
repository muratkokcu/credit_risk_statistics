# Karar Zekâsı — Tam Karar Çerçevesi

**Otomotiv Kredi Finansmanı · Türkiye Bağlamı**

> Beklenen zarar tek başına eksik bir tablo.  
> Çünkü sadece **aşağı riski** ölçüyor, **yukarı fırsatı** görmüyor.

---

## Temel Denklem

```
Net Değer = Beklenen Gelir − Beklenen Zarar − Operasyon Maliyeti
```

Mevcut pipeline yalnızca **ortadaki terimi** (Beklenen Zarar) hesaplıyor.  
Tam karar çerçevesi üç terimin tamamını kapsar.

---

## Katman 1 — Mevcut Pipeline (v1)

### PD · Temerrüt Olasılığı
**Cevapladığı soru:** Bu müşteri önümüzdeki 12 ayda temerrüde düşer mi?

Model, geçmiş kredi verisiyle eğitilmiş bir lojistik regresyon (veya Gradient Boosting) modelinin çıktısıdır.  
Her başvuruya 0–1 arası bir skor verir.

```
logit = intercept
        + (katsayı_findeks   × Findeks skoru)
        + (katsayı_taksit    × taksit/gelir oranı)
        + (katsayı_pesinat   × peşinat oranı)
        + (kategorik düzeltmeler: çalışma tipi, bölge, araç segmenti...)

PD = 1 / (1 + e^(−logit))
```

---

### LGD · Temerrütte Kayıp Oranı
**Cevapladığı soru:** Temerrüt olursa, araç teminatından sonra gerçekte ne kadar kaybederiz?

```
LGD = 1 − Recovery Rate
```

Recovery rate: araç satışından tahsil edilen tutar / temerrüt anındaki bakiye.  
Araç marka kademesi, ikinci el durumu ve bölgesel ikinci el piyasa likiditesine göre segment bazında farklılaşır.

| Segment | Ort. LGD (demo) |
|---------|----------------|
| Yerli · sıfır | %29.5 |
| Yerli · 2. el | %29.6 |
| İthal ekonomik · sıfır | %32.7 |
| İthal ekonomik · 2. el | %33.1 |
| İthal premium · sıfır | %37.6 |
| İthal premium · 2. el | %38.1 |

---

### EAD · Temerrüt Anındaki Risk Tutarı
**Cevapladığı soru:** Müşteri temerrüde düştüğünde üzerimizde ne kadar para kalır?

```
EAD = Finanse Edilen Tutar × CCF
```

**CCF (Credit Conversion Factor):** Geçmiş temerrütlere bakarak "müşteriler temerrüde düştüklerinde borcunun ortalama kaçta kaçı hâlâ ödenmemiş oluyor?" sorusunun cevabı.

Demo modelinde **ortalama CCF = 0.63** → temerrüt anında borcun %63'ü hâlâ ödenmemiş.

CCF'yi etkileyen faktörler:
- **Vade:** Kısa vadeli kredilerde CCF daha yüksek (henüz az ödeme yapılmış)
- **Araç segmenti:** Hafif ticari araçlar genellikle daha erken temerrüde düşüyor
- **Ekonomik döngü:** Kriz dönemlerinde erken temerrüt artar, CCF yükselir

---

### EL · Beklenen Zarar
**Cevapladığı soru:** Bu portföyden önümüzdeki dönemde toplam kaç TL zarar beklemeliyiz?

```
EL = PD × LGD × EAD
```

Portföy genelinde:
```
Toplam EL = Σ (PDᵢ × LGDᵢ × EADᵢ)  ∀ kredi i
```

Demo sonuçları (10.000 başvuru, sentetik veri):
- Toplam finanse edilen tutar: **1.39 milyar TL**
- Toplam beklenen zarar: **136.5 milyon TL**
- Portföy EL oranı: **%9.81**

---

### What-if Simülatörü
**Cevapladığı soru:** Koşulları değiştirirsek EL nasıl değişir?

```python
# Örnek: peşinat artışının etkisi
what_if(profil, change={'down_payment_ratio': 0.35})
```

| Senaryo | PD | EL (TL) | EL Δ |
|---------|-----|---------|------|
| Baz | %89.7 | 135.179 | — |
| Peşinat %15→%35 | %82.5 | ~88.000 | −35% |
| Findeks 1100→1600 | %74.2 | ~76.000 | −44% |
| İkisi birden | %61.8 | ~52.000 | −62% |

---

### Decomposition · Mix / Rate / Volume Ayrıştırması
**Cevapladığı soru:** Yenileme oranı düştüğünde bu düşüş segment mi değişti, performans mı kötüleşti, hacim mi azaldı?

```
Toplam değişim = Mix etkisi + Rate etkisi + Volume etkisi
```

- **Mix etkisi:** Müşteri segmentlerinin ağırlığı değişti mi?
- **Rate etkisi:** Aynı segmentte gerçek performans kötüleşti mi?
- **Volume etkisi:** Toplam hacim arttı mı azaldı mı?

---

### Cohort Analizi
**Cevapladığı soru:** Farklı dönemlerde kredi verdiğimiz müşteriler aynı hızda mı kayboluyor?

Genel oran sağlık yanılsaması yaratır; cohort analizi o yanılsamanın altındaki gerçek trendi gösterir ve sorun büyümeden önce yakalar.

---

### Survival Analizi · Kaplan-Meier + Cox PH
**Cevapladığı soru:** Müşteri ne zaman churn'e düşer ve hangi segment en uzun süre aktif kalır?

Lojistik regresyon "yapacak mı?" sorusunu yanıtlarken, survival analizi "ne zaman?" sorusunu yanıtlar.

---

### SHAP · Model Açıklanabilirliği
**Cevapladığı soru:** Model bu müşteriye neden bu skoru verdi?

```
SHAP değeri = Bu faktörün bu müşterinin skorunu ne kadar artırdığı / azalttığı
```

| Görsel tipi | Ne gösterir |
|-------------|------------|
| Waterfall | Tek müşteri — adım adım skor oluşumu |
| Beeswarm | Tüm portföy — faktör dağılımları |
| Bar | Global önem sıralaması |
| Force plot | Tek müşteri — kısa özet |

---

### Model İzleme · PSI + AUC Zaman Serisi
**Cevapladığı soru:** Model hâlâ güvenilir mi, yoksa yeniden eğitme zamanı geldi mi?

**PSI (Population Stability Index):**
| PSI | Yorum |
|-----|-------|
| < 0.10 | Stabil ✅ |
| 0.10 – 0.25 | Dikkat ⚠️ |
| > 0.25 | Yeniden eğit 🔴 |

**AUC eşikleri:**
| AUC | Yorum |
|-----|-------|
| < 0.65 | Kullanılamaz |
| 0.65 – 0.70 | Zayıf |
| 0.70 – 0.75 | Kabul edilebilir |
| > 0.75 | İyi |

---

### EV · Beklenen Değer Önceliklendirmesi
**Cevapladığı soru:** Satış ekibi bugün kimi aramalı?

```
EV = P(kabul) × Potansiyel Marj − Temas Maliyeti
```

Satış ekibine her sabah "bugün en yüksek beklenen değerli N müşteriyle temas et" listesi üretir.  
Karar intuition değil, model verir.

---

## Katman 2 — Genişletilmiş Çerçeve (Sonraki Adımlar)

### Beklenen Kârlılık · Expected Profitability
**Cevapladığı soru:** Bu krediyi versek gerçekte kârlı mı oluruz?

```
Net Kredi Kârlılığı =
    Faiz geliri
  + Komisyon
  + Sigorta katkısı
  − Fonlama maliyeti
  − Operasyon maliyeti
  − Beklenen zarar (EL)
```

Beklenen zarar düşük ama marj da düşükse vermemek daha mı iyi?  
Beklenen zarar yüksek ama marj çok yüksekse doğru fiyatla vermek kârlı mı?  
Bu soruları ancak kârlılık modeli yanıtlar.

**Öncelik:** Yüksek · Doğrudan kâra dokunuyor.

---

### Optimal Fiyatlama Modeli
**Cevapladığı soru:** Bu müşteriye minimum hangi faizle kredi vermeliyiz?

```
Minimum kabul edilebilir faiz =
    Fonlama maliyeti
  + Operasyon maliyeti
  + EL oranı (PD × LGD)
  + Hedef kâr marjı
```

Aynı faizi herkese vermek ya iyi müşteriyi kaçırır (fazla pahalı) ya da kötü müşteriyi içeri alır (fazla ucuz).  
Fiyatlama modeli her başvuruya "bu müşteri için kırılım faizi şudur, altında verme" der.

**Öncelik:** Yüksek · Hem riski hem kârlılığı optimize ediyor.

---

### RAROC · Risk-Adjusted Return on Capital
**Cevapladığı soru:** Bu portföy için bağladığım sermayenin getirisi nedir?

```
RAROC = Net Kâr / Ekonomik Sermaye
```

Bayi bazında veya segment bazında RAROC hesaplarsan "hangi bayiyle çalışmak gerçekten değerli" sorusuna net cevap verirsin.  
Alternatif kullanıma kıyasla sermayenin doğru yerde bağlı olup olmadığını gösterir.

**Öncelik:** Orta · Yönetim raporlamasında güçlü etki.

---

### Konsantrasyon Riski Analizi
**Cevapladığı soru:** Portföy tek bir noktaya çok mu bağımlı?

Beklenen zarar her krediyi bağımsız değerlendiriyor. Ama aynı bayi, aynı segment, aynı bölgeden 200 kredi varsa — o segment sarsıldığında hepsi aynı anda batabilir.

İzlenmesi gereken konsantrasyon boyutları:
- Bayi başına portföy payı
- Bölge başına EL yoğunluğu
- Araç segmenti başına EAD
- Tek bir marka/model riski (araç değer riski)

**Öncelik:** Orta · Risk yönetimi raporlamasında kritik.

---

### Stres Testi
**Cevapladığı soru:** Kriz olursa portföy ne kadar dayanır?

Türkiye bağlamında kritik senaryolar:

| Senaryo | Etki kanalı |
|---------|------------|
| Döviz %30 değer kaybı | İthal araç fiyatları artar → LGD bozulur |
| Faiz %20 artışı | Taksit/gelir oranı yükselir → PD artar |
| İşsizlik +3 puan | Esnaf/KOBİ segmenti önce erir → PD artar |
| İkinci el piyasa daralması | Recovery rate düşer → LGD artar |

```
Stres EL = Σ (Stres PD × Stres LGD × EAD)
Ek sermaye ihtiyacı = Stres EL − Normal EL
```

**Öncelik:** Orta-Yüksek · Türkiye makro oynaklığı göz önünde bulundurulursa kritik.

---

### CLV · Müşteri Yaşam Boyu Değeri
**Cevapladığı soru:** Bu müşteri uzun vadede bize ne kadar değer yaratır?

```
CLV = Σ (dönem net kârı × yenileme olasılığı × indirim faktörü)
      t=1 → T
```

Beklenen zarar tek bir kredi döngüsüne bakıyor. CLV tüm ilişkiye bakıyor.

**Kullanım alanları:**
- "Şimdi sıfır marjla versek bile 5 yılda değer yaratır mı?"
- Bayi kalite skoru: hangi bayi uzun vadeli değerli müşteri gönderiyor?
- Churn önleme önceliği: CLV yüksek müşteri erken uyarı sistemine önce girer

**Öncelik:** Yüksek · EV modeliyle birleşince en güçlü karar aracına dönüşür.

---

## Tam Karar Çerçevesi — Özet Tablo

| Soru | Model | Katman |
|------|-------|--------|
| Bu müşteri batacak mı? | PD | v1 ✅ |
| Batarsa ne kadar kaybederiz? | LGD × EAD = EL | v1 ✅ |
| Koşul değiştirirsek ne olur? | What-if simülatörü | v1 ✅ |
| Düşüş nereden kaynaklanıyor? | Decomposition | v1 ✅ |
| Yeni müşteriler mi daha kötü? | Cohort analizi | v1 ✅ |
| Ne zaman churn'e düşer? | Survival analizi | v1 ✅ |
| Model kararını nasıl açıklarım? | SHAP | v1 ✅ |
| Model hâlâ güvenilir mi? | PSI + AUC izleme | v1 ✅ |
| Bugün kimi arayalım? | EV önceliklendirme | v1 ✅ |
| Versek kârlı mı olur? | Expected Profitability | v2 ⬜ |
| Doğru faiz nedir? | Optimal Fiyatlama | v2 ⬜ |
| Sermayemi verimli kullanıyor muyum? | RAROC | v2 ⬜ |
| Portföy çok mu konsantre? | Konsantrasyon analizi | v2 ⬜ |
| Kriz olursa ne olur? | Stres testi | v2 ⬜ |
| Bu müşteri uzun vadede değerli mi? | CLV | v2 ⬜ |

---

## v2 Geliştirme Öncelik Sırası

**1. Optimal Fiyatlama** — hem riski hem kârlılığı aynı anda optimize ediyor, en hızlı geri dönüşlü alan.

**2. CLV + EV birleşimi** — "kimi arayalım" sorusunu "kısa vadeli kazanç" değil "uzun vadeli değer" eksenine taşıyor.

**3. Stres testi** — Türkiye makro oynaklığı göz önünde bulundurulursa yönetim raporlamasında güçlü etki.

**4. Expected Profitability** — fiyatlama modeli kurulduktan sonra doğal devam adımı.

**5. RAROC + Konsantrasyon** — kurumsal raporlama ve bayi yönetimi için.

---

*Bu doküman v1 pipeline çıktılarını ve v2 genişletme planını kapsar.*  
*Tüm sayısal örnekler sentetik veriyle üretilmiştir.*
