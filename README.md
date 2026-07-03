# Karar Zekâsı — Analitik Pipeline

**Otomotiv Kredi Finansmanı · Türkiye Bağlamı**  
Raporlamayı *"ne oldu"*dan *"ne yapmalıyım"*a taşıyan tam analitik pipeline.

---

## Klasör Yapısı

```
credit_risk/
├── README.md                    # Bu dosya
├── requirements.txt             # Python bağımlılıkları
├── notebooks/                   # Jupyter notebook'lar
│   ├── karar_zekasi_tam_pipeline.ipynb   # Ana pipeline — 12 bölüm
│   └── credit_risk_pd_lgd_ead_demo.ipynb # Eski PD/LGD/EAD demosu
├── docs/                        # Doküman ve soru-cevap notları
│   ├── karar_cercevesi_tam.md
│   └── kredi_risk_modelleme_v1_soru_cevap.md
├── figures/                     # Notebook'un ürettiği PNG grafikleri
└── app/                         # Standalone web simülatörü
    └── kredi_risk_simulatoru.html
```

### Notebook bölümleri

| # | Bölüm | Analitik Seviye |
|---|-------|-----------------|
| 0 | Bağımlılıklar ve ayarlar | — |
| 1 | Sentetik veri üretimi (Türkiye otomotiv şeması) | — |
| 2 | PD Modeli — Lojistik Regresyon + Gradient Boosting | L3 |
| 3 | LGD Modeli — Segment bazlı recovery tahmini | L3 |
| 4 | EAD / CCF Modeli | L3 |
| 5 | Expected Loss (EL = PD × LGD × EAD) | L3 |
| 6 | What-if Simülatörü | L4 |
| 7 | Decomposition Analizi (Mix / Rate / Volume) | L2 |
| 8 | Cohort Analizi | L2 |
| 9 | Survival Analizi (Kaplan-Meier + Cox PH) | L3 |
| 10 | SHAP Değerleri — Model açıklanabilirliği | L3 |
| 11 | Model İzleme — PSI + AUC zaman serisi | L5 |
| 12 | Beklenen Değer (EV) Önceliklendirmesi | L4 |

---

## Hızlı Başlangıç

### Ön koşul

- Python 3.10 veya üzeri
- WSL2 / Ubuntu / macOS / Linux (Windows'ta doğrudan da çalışır)

---

### 1 — Repoyu klonla veya klasörü oluştur

```bash
mkdir karar-zekasi && cd karar-zekasi
# notebook ve requirements.txt dosyalarını bu klasöre koy
```

---

### 2 — Sanal ortam oluştur

```bash
python3 -m venv venv
source venv/bin/activate        # Linux / macOS / WSL2
# Windows CMD için: venv\Scripts\activate
```

> **Neden venv?**  
> Sistem Python'unu kirletmemek için. Ubuntu 22+ ve macOS Sonoma+
> doğrudan `pip install` yapmana izin vermiyor
> (`externally-managed-environment` hatası). `venv` bu sorunu çözer.

---

### 3 — Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

Beklenen süre: ~2-4 dakika (shap ve lifelines derleme gerektirir).

---

### 4 — JupyterLab'ı başlat

```bash
jupyter lab
```

Tarayıcı otomatik açılmazsa terminaldeki `http://localhost:8888/lab?token=...`
adresini kopyala ve tarayıcıya yapıştır.

---

### 5 — Notebook'u çalıştır

JupyterLab'da sol panelden `notebooks/karar_zekasi_tam_pipeline.ipynb` dosyasını aç.

**Tüm hücreleri çalıştırmak için:**  
`Run` menüsü → `Run All Cells`

**Tek tek ilerlemek için:**  
Her hücrede `Shift + Enter`

**Kernel sıfırlamak için (hata alırsan):**  
`Kernel` menüsü → `Restart Kernel and Run All Cells`

---

## Çıktılar

Notebook çalıştıktan sonra aşağıdaki PNG dosyaları `figures/` klasöründe oluşur:

| Dosya | İçerik |
|-------|--------|
| `pd_model.png` | Katsayı grafiği + ROC eğrisi |
| `lgd_segment.png` | Segment bazlı LGD ısı haritası |
| `ead_ccf.png` | CCF dağılımı + vade grafiği |
| `expected_loss.png` | Bölge bazlı EL + dağılım |
| `what_if.png` | Senaryo karşılaştırma grafiği |
| `decomposition.png` | Mix/Rate/Volume waterfall |
| `cohort.png` | Cohort yenileme eğrileri |
| `survival.png` | Kaplan-Meier + Cox PH |
| `shap_beeswarm.png` | SHAP beeswarm dağılımı |
| `shap_bar.png` | Global feature importance |
| `shap_waterfall.png` | Tek müşteri waterfall |
| `model_monitoring.png` | AUC + PSI zaman serisi |
| `ev_prioritization.png` | Beklenen değer bubble chart |

---

## Sık Karşılaşılan Sorunlar

### `ModuleNotFoundError: No module named 'numpy'`
Sanal ortam aktif değil. Çözüm:
```bash
source venv/bin/activate
```

### `externally-managed-environment` hatası (pip)
`venv` olmadan sistem Python'una kurmaya çalışıyorsun.  
2. adımdaki `venv` talimatlarını takip et.

### JupyterLab token istiyor
Terminalde `jupyter lab` çıktısındaki tam URL'yi (token dahil) tarayıcıya yapıştır.  
Ya da şifre belirle:
```bash
jupyter server password
```

### SHAP yavaş çalışıyor
Bölüm 10'da `X_test[:500]` ile 500 örnek kullanılıyor.  
Daha hızlı çalıştırmak için `[:100]` olarak değiştir.

### `lifelines` kurulumunda C derleyicisi hatası (Windows)
```bash
pip install lifelines --no-build-isolation
```
veya WSL2 kullan — derleyici sorunları orada yaşanmaz.

---

## Production'a Taşıma Yol Haritası

### v1 → v2: Gerçek veri bağlantısı

```python
# Bölüm 1'deki sentetik veri bloğunu şununla değiştir:
import oracledb

conn = oracledb.connect(
    user="KREDI_USER",
    password=os.environ["DB_PASSWORD"],
    dsn="host:port/service"
)
df = pd.read_sql("SELECT * FROM KREDI_BASVURU WHERE ...", conn)
```

### v2 → v3: FastAPI endpoint

What-if simülatörünü (Bölüm 6) bir API endpoint'ine sar:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Basvuru(BaseModel):
    findeks_score: int
    down_payment_ratio: float
    # ... diğer alanlar

@app.post("/risk/hesapla")
def hesapla(b: Basvuru):
    return what_if(b.dict())
```

### v3 → v4: Otomatik model izleme

PSI > 0.25 olduğunda (Bölüm 11) yeniden eğitim pipeline'ını tetikle:

```python
if psi > 0.25:
    retrain_model()
    send_alert("Model yeniden eğitildi — PSI: {psi:.3f}")
```

---

## Teknik Notlar

- Tüm veriler sentetiktir; gerçek müşteri verisi içermez.
- Findeks/KKB skoru gerçek üretimde ilgili API'den çekilecektir.
- LGD modeli gerçek tahsilat/icra verisiyle kalibre edilmelidir.
- PSI eşikleri endüstri standardına göredir (Basel IRB rehberi).

---

## Bağımlılık Sürümleri

```
Python     3.12.3
numpy      2.4.4
pandas     2.3.3
matplotlib 3.10.8
seaborn    0.13.2
scikit-learn 1.8.0
shap       0.52.0
lifelines  0.30.3
jupyterlab 4.3.6
```
# credit_risk_statistics
