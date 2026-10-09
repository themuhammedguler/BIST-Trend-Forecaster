---
title: BIST Trend Forecaster
emoji: 📈
colorFrom: green
colorTo: blue
sdk: streamlit
sdk_version: 1.39.0
app_file: app.py
pinned: false
---

# 📈 BIST Trend Forecaster (AI-Based Stock Prediction)

Bu proje, **MultiGroup Zero2End Machine Learning Bootcamp** bitirme projesi olarak geliştirilmiştir. BIST 30 hisselerinin geçmiş verilerini ve teknik indikatörleri kullanarak, bir sonraki işlem gününde hissenin **Yükseleceğini mi** yoksa **Düşeceğini/Yatay kalacağını mı** tahmin eder.

### 🖥️ Canlı Demo ve Uygulama

BIST 30 hisseleri için geliştirdiğimiz yapay zeka modelini canlı verilerle test edebilirsiniz. Kurulum yapmanıza gerek yoktur.

| Platform | Durum | Link |
| :--- | :---: | :--- |
| **Streamlit Cloud** | 🟢 Aktif | [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bist-trend-forecaster.streamlit.app/) |
| **Hugging Face** | 🟢 Aktif | [![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/themuhammedguler/BIST-Trend-Forecaster) |

## 1. Problem Tanımı
Finansal piyasalarda bireysel yatırımcılar genellikle teknik analiz yapmakta zorlanır ve duygusal kararlar verirler.
*   **Problem:** Karmaşık teknik göstergelerin yorumlanmasının zorluğu ve piyasa gürültüsü içinde doğru sinyali bulamama.
*   **Çözüm:** Geçmiş fiyat hareketlerini ve teknik indikatörleri (RSI, MACD, SMA vb.) analiz ederek matematiksel bir "Yön Tahmini" (Binary Classification) sunan bir Makine Öğrenmesi modeli.

## 2. Veri Seti ve Hazırlık
*   **Veri Kaynağı:** `yfinance` kütüphanesi ile Yahoo Finance üzerinden çekilmiştir.
*   **Kapsam:** BIST 30 endeksindeki 30 şirketin son 8 yıllık (2018-2025) günlük verileri.
*   **Veri Büyüklüğü:** Yaklaşık 70.000+ satır (PDF gereksinimi olan 10k satır fazlasıyla karşılanmıştır).
*   **Feature Engineering (Öznitelik Mühendisliği):**
    *   RSI (14), MACD, Bollinger Bands
    *   SMA (10 ve 50 günlük hareketli ortalamalar)
    *   Volatilite ve Momentum (Lag Features)
    *   Takvim Etkisi (Haftanın günü, Ayın günü)

## 3. Modelleme Süreci
### Baseline Model
*   Başlangıçta "Yarın, bugünün aynısıdır" mantığıyla basit bir yaklaşım test edildi. Başarı oranı %50 civarındaydı (Rastgele tahmin).

### Final Model: XGBoost
*   Tabular verilerde yüksek performans gösterdiği için **XGBoost Classifier** seçildi.
*   **Validasyon Şeması:** Finansal verilerde "geleceği görmeyi" (look-ahead bias) ve hisseler arası gün içi sızıntıyı engellemek için **Tarihsel Zaman Kesimi (`get_temporal_split`)** kullanıldı. Belirlenen tarihe kadarki tüm hisse verileri eğitim setine, o tarih ve sonrasındaki dönem ise test setine ayrıldı.

### Hiperparametre Optimizasyonu: Optuna
*   `src/tune.py`, XGBoost hiperparametrelerini **Optuna** (TPE sampler + MedianPruner) ile arar. Her deneme, tarihe göre ileri doğru (walk-forward) zaman serisi CV'si ile değerlendirilir; aynı günün satırları hep aynı katta kalır.
*   İlk deneme mevcut manuel parametrelerle (`n_estimators=100, learning_rate=0.05, max_depth=5`) başlar.
*   Kullanım (varsayılan eğitim davranışı değişmez):
    ```bash
    cd src
    python model_train.py                        # manuel parametreler
    python model_train.py --tune --n-trials 50   # Optuna ile optimizasyon
    python model_train.py --tune --metric roc_auc # CV hedefi: accuracy | balanced_accuracy | roc_auc
    python model_train.py --balance-classes      # sınıf oranına göre scale_pos_weight
    ```

### Değerlendirme Metrikleri
*   Doğruluk tek başına yanıltıcıdır: çoğunluk sınıfını tahmin eden bir model yüksek doğruluk alır ama hiçbir şey öğrenmemiş olur. Bu yüzden `model_train.py`, eğitim/test setlerinin sınıf dağılımını loglar ve doğruluğun yanında **Dengeli Doğruluk (Balanced Accuracy)**, **ROC-AUC**, **Log Loss** ve **Kesinlik (Precision)** değerlerini raporlar (`src/metrics.py`).
*   `--balance-classes`, yükseliş/düşüş oranı dengesizleştiğinde XGBoost'un `scale_pos_weight` parametresini eğitim etiketlerinden hesaplar; Optuna CV'sinde her kat kendi eğitim etiketlerini kullanır.

### Testler
Geliştirme ve test bağımlılıklarını kurup testleri çalıştırabilirsiniz:
```bash
pip install -r requirements-dev.txt
pytest -q
```

### Kod Kalitesi
Kod stili ve lint kuralları `pyproject.toml` içinde tanımlıdır ([ruff](https://docs.astral.sh/ruff/); PEP 8, Pyflakes ve import sırası, satır sınırı 100). CI her push ve PR'da aynı kontrolleri çalıştırır:
```bash
ruff check .          # lint (otomatik düzeltme için: ruff check --fix .)
ruff format .         # biçimlendirme (yalnızca kontrol için: ruff format --check .)
```
Toplu biçimlendirme commit'ini `git blame` çıktısında gizlemek için: `git config blame.ignoreRevsFile .git-blame-ignore-revs`

### Model Performansı
*   **Test Seti (varsayılan parametreler, kesim 2025-02-25):** Doğruluk 0.511, Dengeli Doğruluk 0.511, ROC-AUC 0.513, Log Loss 0.698. Test setindeki yükseliş günü oranı %49.4'tür.
    *   *Yorum:* Finansal piyasaların stokastik yapısı göz önüne alındığında sinyal zayıftır; ROC-AUC'nin 0.5'e yakın olması tahminlerin çoğunun yazı-tura seviyesinde olduğunu gösterir. Bu nedenle arayüzde nötr eşik bandı kullanılır.
*   **Önemli Öznitelikler:** Model kararlarında en çok `day_of_week` (haftanın günü), `month` (ay) ve `vol_change` (hacim değişimi) etkili olmuştur.

### Model Mimarileri Kıyaslama (Benchmark)
Farklı makine öğrenimi ve derin öğrenme mimarilerinin BIST 100 yön tahminindeki performansını sızıntısız (leak-free) ve adil bir şekilde karşılaştırmak için `src/benchmark.py` yürütülür:
*   **Değerlendirme:** 3 katmanlı genişleyen pencereli ileriye dönük zaman serisi çapraz doğrulama (walk-forward expanding window cross-validation).
*   **Karşılaştırılan Modeller:**
    *   *Temel Referanslar (Baselines):* Always Down, Always Up, Repeat Yesterday, Regularized Logistic Regression.
    *   *Gradient Boosting Varyantları:* XGBoost, LightGBM, Scikit-learn HistGradientBoosting, Calibrated XGBoost (Platt Scaling).
    *   *Derin Öğrenme (Deep Learning):* Multi-Layer Perceptron (MLP), PyTorch GRU yinelemeli ağ (Recurrent Neural Network).
*   **Kullanım:**
    ```bash
    python src/benchmark.py --output-report docs/experiments/model_benchmark_report.md
    ```
*   **Sonuç:** Ayrıntılı ampirik bulgular, Sharpe oranı ve kümülatif strateji getirileri için [docs/experiments/model_benchmark_report.md](docs/experiments/model_benchmark_report.md) raporunu inceleyebilirsiniz.

## 4. İş Gereksinimleri ve Kullanım
Bu model, bir yatırım tavsiyesi vermekten ziyade, yatırımcının karar destek mekanizması olarak tasarlanmıştır.
*   **Canlıya Alma:** Model, `Streamlit` kullanılarak interaktif bir web arayüzüne dönüştürülmüştür.
*   **İzleme (Monitoring):** Canlı ortamda modelin başarısı "Doğru Yön Tahmini Yüzdesi" metriği ile haftalık olarak takip edilmelidir.

## 5. Proje Yapısı
```text
BIST-TREND-FORECASTER/
├── .github/workflows/     # CI, CD ve periyodik yeniden eğitim iş akışları
├── data/                  # Ham ve işlenmiş veriler
├── docs/experiments/      # Model mimarisi kıyaslama ve deney raporları
├── models/                # Eğitilmiş .json modelleri ve model meta verileri
├── notebooks/             # EDA ve Deneme not defterleri
├── src/                   # Kaynak kodlar
│   ├── benchmark.py       # Genişleyen pencereli model benchmark harness
│   ├── config.py          # Ayarlar ve özellik listeleri
│   ├── features.py        # Teknik indikatör ve makro özellik hesaplamaları
│   ├── scanner.py         # BIST 30 piyasa taraması ve fırsat radarı
│   ├── tune.py            # Optuna hiperparametre optimizasyonu
│   ├── metrics.py         # Değerlendirme metrikleri ve sınıf ağırlıklandırma
│   ├── model_metadata.py  # Model üretim meta verisi kayıt modülü
│   └── model_train.py     # Model eğitimi ve kalite kapısı (quality gate)
├── tests/                 # Birim ve entegrasyon testleri
├── app.py                 # Streamlit arayüz kodu
├── requirements.txt       # Kütüphane bağımlılıkları (prod)
├── requirements-dev.txt   # Geliştirme ve test bağımlılıkları
├── pyproject.toml         # Proje meta verisi ve ruff ayarları
└── README.md              # Proje dokümantasyonu
```
