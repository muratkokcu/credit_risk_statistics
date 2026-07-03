# Kredi Risk Modelleme — v1 Soru & Cevap Kağıdı

**Konu:** PD / LGD / EAD tabanlı Expected Loss modeli — otomotiv kredi finansmanı demo'su
**Versiyon:** v1 (sentetik veri, tek seferlik notebook)
**Format:** Sınav kağıdı — soru, ardından cevap

---

## Bölüm 1 — Temel Kavramlar

**S1. PD nedir?**
C: Probability of Default — Temerrüt Olasılığı. Bir müşterinin belirli bir zaman diliminde (genelde 12 ay) kredisini ödeyemez hale gelme olasılığı. 0 ile 1 arasında bir skordur.

**S2. LGD nedir?**
C: Loss Given Default — Temerrütte Kayıp Oranı. Müşteri temerrüde düşerse, kurumun batırdığı tutarın krediye oranı (yüzde). Teminat varsa (örneğin araç) ve değeri yüksekse LGD düşer; teminatsız veya teminat değeri düşükse LGD yükselir.

**S3. EAD nedir?**
C: Exposure at Default — Temerrüt Anındaki Risk Tutarı. Müşteri temerrüde düştüğü anda kurumun o krediden maruz kaldığı toplam tutar (ödenmemiş anapara bakiyesi).

**S4. Expected Loss (EL) formülü nedir?**
C: EL = PD × LGD × EAD. Üç bileşen çarpılarak, tek bir kredi için beklenen parasal zarar bulunur. Portföy genelinde tüm kredilerin EL'leri toplanınca toplam beklenen zarar ortaya çıkar.

**S5. PD, LGD ve EAD neden ayrı ayrı modellenir, neden tek bir model değil?**
C: Çünkü her biri farklı bir soruya cevap verir ve farklı aksiyon alanlarına karşılık gelir — "temerrüde düşer mi" (PD) müşteri seçimi/onay politikasıyla, "düşerse ne kadar kaybederiz" (LGD) teminat politikasıyla, "ne kadar bakiye kalır" (EAD) vade/ödeme yapısıyla ilgilidir. Ayrıştırılmış olmaları, hangi değişkenin senaryoyu nasıl etkilediğini izole etmeyi sağlar (what-if mantığının temeli budur).

**S5b. PD somut olarak nasıl hesaplanıyor — denklemler nedir?**
C: İki adımda hesaplanıyor.

*Adım 1 — Logit (lojistik regresyon denklemi):* Her değişken, modelin eğitimde öğrendiği bir katsayı ile çarpılıp toplanır.

```
logit = intercept
        + (katsayı_findeks   × Findeks skoru)
        + (katsayı_taksit    × taksit/gelir oranı)
        + (katsayı_pesinat   × peşinat oranı)
        + (katsayı_krediler  × mevcut kredi sayısı)
        + (katsayı_yas       × müşteri yaşı)
        + (kategorik düzeltmeler: çalışma tipi, araç segmenti, marka kademesi, bölge)
```

Demo modelindeki gerçek katsayılarla:

```
logit = 4,12
        + (-0,00337 × Findeks skoru)
        + (2,408    × taksit/gelir oranı)
        + (-1,579   × peşinat oranı)
        + (0,314    × mevcut kredi sayısı)
        + (-0,00142 × müşteri yaşı)
        + Esnaf/KOBİ:        +0,559   |  Serbest meslek: +0,179   |  Ücretli: +0,028
        + SUV:               +0,076   |  Hafif Ticari:   -0,218   |  Binek-Orta: +0,014
        + İthal-Ekonomik:    +0,173   |  İthal-Premium:  +0,238
        + Marmara:           +0,118   |  Doğu Anadolu:   +0,371   |  Güneydoğu: +0,258
                                       |  Ege:            -0,129   |  Karadeniz: -0,041
```

Pozitif katsayı riski artırır, negatif katsayı azaltır (örnek: peşinat katsayısı -1,579 — peşinat arttıkça logit düşer, risk azalır).

*Adım 2 — Sigmoid (lojistik) fonksiyon — logit'i 0-1 arası olasılığa çevirir:*

```
PD = 1 / (1 + e^(-logit))
```

*Sayısal örnek (Findeks 1250, peşinat %15, mevcut kredi 2, yaş 34, Esnaf/KOBİ, SUV, İthal-Ekonomik, Marmara, taksit/gelir oranı 0,234):*

```
logit = 4,12 - 4,21 + 0,564 - 0,237 + 0,628 - 0,048 + 0,559 + 0,076 + 0,173 + 0,118
      ≈ 1,706

PD = 1 / (1 + e^(-1,706)) ≈ 0,846  →  %84,6
```

*Kısa sözel özet:* PD, geçmiş kredi verisiyle eğitilmiş bir lojistik regresyon modelinin çıktısıdır. Model her başvuru özelliğine bir ağırlık (katsayı) atar, bunları toplayıp tek bir skora (logit) indirger, sonra bu skoru sigmoid fonksiyonuyla 0-1 arasında bir olasılığa dönüştürür. Katsayılar, geçmişte kimlerin gerçekten temerrüde düştüğü verisinden (maximum likelihood yöntemiyle) öğrenilir.

---

## Bölüm 2 — Referans Repo Değerlendirmeleri

**S6. Loan-amount-prediction-regression (semasuka) projesi ne ölçüyor, neden yetersiz bulundu?**
C: Kredi tutarını (limit) tahmin eden bir regresyon modeli — onay/red değil, "ne kadar verilecek" sorusuna cevap arıyor. Yetersiz bulunma nedenleri: RMSE'nin (~20.800 TL, 0-400.000 aralığında) belirsizlik payı yüksek; sadece nokta tahmini sunuyor, kısmi bağımlılık grafiği (partial dependence) veya SHAP gibi gerçek what-if analiz araçları yok; veri seti meta-veri eksikliği taşıyor (gider tipi, property type gibi alanlar tanımsız); kurumsal ölçekte referans mimari değil.

**S7. credit_risk (brunokatekawa) projesi ne ölçüyor?**
C: PD (temerrüt olasılığı) — sınıflandırma problemi. CatBoost Classifier kullanmış, kalibrasyon yapmış (olasılıkları gerçek olasılık gibi yorumlanabilir hale getirmiş), PD'yi LGD (sabit %100 varsayım) ve EAD (kredi tutarı) ile çarpıp portföy bazında toplam Expected Loss hesaplamış (~17,8 milyon dolar tasarruf gösterilmiş, modelsiz duruma kıyasla).

**S8. Bu iki repodan hangisi senin projen için daha sağlam bir referans, neden?**
C: İkincisi (credit_risk / brunokatekawa) daha sağlam. Çünkü PD-LGD-EAD üçlüsünü ayrıştırıp iş etkisini (dolar bazlı beklenen zarar) doğrudan modele bağlamış; bu yapı NPL/FPD risk dashboard'una ve renewal opportunity score işine doğal oturuyor.

**S9. Türkiye'ye özgü, kamuya açık, kaliteli bir PD/LGD/EAD repo'su var mı?**
C: Pratik olarak yok denecek kadar az. Bu alan Türkiye'de büyük ölçüde bankaların/finans şirketlerinin kapalı kapılar ardında, danışmanlık firmaları (RiskActive, RDC gibi) üzerinden yürüttüğü bir iş; açık kaynak kültürü zayıf.

**S10. Türkiye'de bankalar Basel IRB (içsel derecelendirme) yaklaşımını kullanabiliyor mu?**
C: Hayır. BDDK'dan henüz hiçbir banka IRB onayı almamış olduğu için tüm bankalar Standart Yaklaşım'ı kullanarak risk ağırlıklı varlıkları (RWA) hesaplıyor. Bu, içsel PD/LGD/EAD modellerinin resmi sermaye yeterliliği hesabında kullanılamadığı anlamına geliyor.

**S11. Bu regülasyon durumunun otomotiv kredi finansman şirketi (banka olmayan) için anlamı nedir?**
C: BDDK'nın sermaye yeterliliği regülasyonu muhtemelen doğrudan bağlamıyor — bu bir avantaj, çünkü PD/LGD/EAD çerçevesi regülatif zorunluluk olmadan, saf içsel risk yönetimi ve karar destek aracı olarak kullanılabilir; Basel'in regülatif RWA formülleri atlanabilir.

---

## Bölüm 3 — Demo Notebook (v1)

**S12. Demo'da kaç başvuru içeren sentetik veri üretildi ve hangi alanlar kullanıldı?**
C: 8.000 başvuru. Alanlar: dealer_id, bölge, araç segmenti, marka kademesi, araç durumu (sıfır/2.el), müşteri yaşı, gelir, çalışma durumu, Findeks/KKB skoru, mevcut kredi sayısı, kredi tutarı, peşinat oranı, vade, finanse edilen tutar, aylık taksit, taksit/gelir oranı.

**S13. Temerrüt etiketi (default_flag) nasıl üretildi?**
C: Gerçekçi bir risk mantığı kurularak (lojistik fonksiyon / logit) simüle edildi: düşük Findeks skoru, yüksek taksit/gelir oranı, düşük peşinat, çok sayıda mevcut kredi, Esnaf/KOBİ çalışma tipi ve 2.el araç, temerrüt olasılığını artıracak şekilde ağırlıklandırıldı; buna rastgele gürültü eklendi.

**S14. PD modeli için hangi algoritma kullanıldı ve sonucu neydi?**
C: Lojistik regresyon (class_weight="balanced"). Test AUC: 0,751 — iyi bir ayırt edici güç.

**S15. PD modelinde riski en çok artıran ve en çok azaltan faktörler nelerdi?**
C: En çok artıran: yüksek taksit/gelir oranı, Esnaf/KOBİ çalışma tipi, Doğu/Güneydoğu Anadolu bölgesi, mevcut kredi sayısı. En çok azaltan: yüksek peşinat oranı, Hafif Ticari araç segmenti, Ege bölgesi.

**S16. LGD modeli nasıl kuruldu ve sonucu neydi?**
C: Sadece default eden krediler üzerinde, recovery_rate'i (1-LGD) tahmin eden bir lineer regresyon; ana değişkenler marka kademesi ve araç durumu (sıfır/2.el). Test R²: -0,013 — yani model, sadece bu sınırlı değişken setiyle recovery oranını anlamlı şekilde açıklayamadı.

**S17. LGD modelinin zayıf çıkması neyi gösteriyor?**
C: Bu aslında gerçekçi bir bulgu — gerçek hayatta da LGD en zor modellenen parametredir. Daha fazla ve daha güçlü açıklayıcı değişkene (icra süreci süresi, teminat değerleme yöntemi, bölgesel ikinci el piyasa likiditesi gibi) ihtiyaç duyulur; sonuç gizlenmedi, olduğu gibi raporlandı.

**S18. EAD modeli (CCF — Credit Conversion Factor) nasıl kuruldu ve sonucu neydi?**
C: Kalan bakiye oranı (outstanding_at_default / financed_amount) hedef değişken olarak alındı; vade, hayatta kalınan ay sayısı ve araç segmenti ile lineer regresyon kuruldu. Test R²: 0,911 — çok güçlü, çünkü büyük ölçüde matematiksel olarak vade/ödenen ay sayısından türetilebilir bir ilişki.

**S19. Portföy bazında toplam Expected Loss ne çıktı?**
C: Toplam finanse edilen tutar ~1,14 milyar TL; toplam beklenen zarar (EL) ~107 milyon TL; portföy EL oranı ~%9,4. Bölge bazında en yüksek EL Marmara'da (~31,9 milyon TL), en düşük Doğu Anadolu'da (~8,9 milyon TL) çıktı (yalnızca hacim farkından kaynaklanıyor, bölgesel risk yoğunluğu ayrıca incelenmeli).

**S20. What-if simülasyonunda hangi senaryolar karşılaştırıldı ve sonuç ne oldu?**
C: Aynı başvuru profili (Esnaf/KOBİ, 2.el SUV, İthal-Ekonomik) üzerinde: Baz senaryo EL ~122.877 TL; peşinatı %15'ten %35'e çıkarınca EL ~80.549 TL'ye (~%34 azalma); Findeks skorunu 1250'den 1600'e çıkarınca EL ~79.857 TL'ye (~%35 azalma); ikisi birlikte uygulanınca EL ~45.176 TL'ye (~%63 azalma) düştü.

---

## Bölüm 4 — Web Simülatörü

**S21. Web simülatöründeki hesaplama mantığı notebook'taki modelden farklı mı?**
C: Hayır. Notebook'ta eğitilen PD modelinin gerçek katsayıları (intercept + her değişkenin coefficient'i), segment bazlı LGD ortalamaları ve ortalama CCF değeri JavaScript'e aktarılıp birebir aynı formülle (logit → sigmoid → PD; segLgd lookup; EAD = financed × avgCcf; EL = PD×LGD×EAD) hesaplanıyor.

**S22. Web simülatöründe hangi değişkenler canlı değiştirilebiliyor?**
C: Findeks skoru, peşinat oranı, aylık gelir, araç bedeli, vade, mevcut kredi sayısı, çalışma durumu, bölge, araç segmenti, marka kademesi, araç durumu (sıfır/2.el), müşteri yaşı.

---

## Bölüm 5 — Sonraki Versiyonlar İçin Yol Haritası

**S23. v1 modelinin temel sınırlılığı nedir?**
C: Tek seferlik, statik bir notebook — sentetik veriyle eğitildi, gerçek Oracle DB verisine bağlı değil, zaman içinde yeniden eğitilmiyor, doğruluğunun zaman içindeki seyri izlenmiyor.

**S24. "Sürekli beslenen model" kredi riski bağlamında neden gerçek zamanlı olamaz?**
C: Çünkü bir kredinin "temerrüde düştü mü düşmedi mi" etiketi ancak aylar sonra (vade boyunca izlenerek) öğrenilebilir. Bu yüzden sistem anlık değil, periyodik ve disiplinli bir yeniden eğitim döngüsü olarak kurgulanmalı.

**S25. İleri versiyon mimarisi hangi dört katmandan oluşuyor?**
C: (1) Veri katmanı — periyodik, versiyonlanmış Oracle DB snapshot'ları + olgunlaşma penceresi tanımlı etiketleme; (2) Model katmanı — zaman bazlı (out-of-time) train/test ayrımı + model versiyon kaydı; (3) Değerlendirme/anlamlandırma katmanı — AUC, KS, Gini, Brier Score zaman serisi takibi + Population Stability Index (PSI) ile veri kaymasının izlenmesi; (4) Sonuç/karar katmanı — model skorlarının gerçekleşen sonuçlarla karşılaştırıldığı backtesting dashboard'u.

**S26. Modelin "gerçekten iyileşip iyileşmediğini" ölçmek için hangi metrikler zaman serisi olarak takip edilmeli?**
C: AUC (ayırt edici güç), KS istatistiği, Gini katsayısı, Brier Score (kalibrasyon kalitesi) ve PSI (popülasyon kayması).

**S27. PSI yükseldiğinde ne anlama gelir, ne yapılmalı?**
C: Veri popülasyonunun zaman içinde kaydığı, modelin artık güncel müşteri profiline uymadığı anlamına gelir; bu durum yeniden eğitim tetikleyicisi olarak kullanılmalı.

**S28. Champion/challenger yaklaşımı nedir?**
C: Yeni eğitilen model (challenger), eski/production'daki model (champion) ile aynı dönem üzerinde paralel test edilir; yeni model sadece gerçekten daha iyi performans gösteriyorsa devreye alınır — kontrolsüz model değişimini önler.

**S29. Önerilen pratik yol haritası (v1 → v4) nedir?**
C: v1 — şu an elimizdeki tek seferlik notebook (sentetik veri). v2 — Oracle DB'den otomatik veri çekme + zamanlanmış (örneğin aylık) yeniden eğitim + her turun AUC/KS/PSI metriklerini bir tabloya kaydetme. v3 — bu metriklerin React dashboard'unda zaman serisi grafiği (model "doğruluk eğrisi") olarak gösterilmesi. v4 — champion/challenger karşılaştırması ve otomatik model değiştirme eşiği.

**S30. Bu yapı, hangi iş alanlarına doğrudan taşınabilir?**
C: Otomotiv kredi finansmanı bağlamındaki NPL/FPD risk dashboard'una ve renewal opportunity score (banka-dealer iki taraflı platform) işine — PD/LGD/EAD ayrıştırması, what-if simülatörü ve sürekli model izleme mimarisi her ikisinde de doğrudan iskelet olarak kullanılabilir.

---

*Not: Bu dokümandaki tüm sayısal sonuçlar (AUC, R², EL tutarları) sentetik veriyle üretilmiş demo çıktılarıdır, gerçek kredi kararı veya yönetim raporlaması için kullanılmamalıdır.*
