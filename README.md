# 📈 Borsa İstanbul (BIST 30) Yapay Zeka Yön Tahmini

Bu proje, **MultiGroup Zero2End Machine Learning Bootcamp** bitirme projesi olarak geliştirilmiştir. BIST 30 hisselerinin geçmiş verilerini ve teknik indikatörleri kullanarak, bir sonraki işlem gününde hissenin **Yükseleceğini mi** yoksa **Düşeceğini/Yatay kalacağını mı** tahmin eder.

🔗 **Canlı Proje Linki:** [BIST Prediction App](https://bist-prediction.streamlit.app/)

---

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
*   **Validasyon Şeması:** Finansal verilerde "geleceği görmeyi" (look-ahead bias) engellemek için klasik K-Fold yerine **`TimeSeriesSplit`** (Zaman Serisi Ayrımı) kullanıldı. İlk yıllar eğitim, son aylar test seti olarak ayrıldı.

### Model Performansı
*   **Doğruluk (Accuracy):** %55 - %60 bandında.
    *   *Yorum:* Finansal piyasaların stokastik yapısı göz önüne alındığında, %50 üzerindeki her oran istatistiksel bir avantaj (edge) sağlar.
*   **Önemli Öznitelikler:** Model kararlarında en çok `day_of_week` (haftanın günü), `month` (ay) ve `vol_change` (hacim değişimi) etkili olmuştur.

## 4. İş Gereksinimleri ve Kullanım
Bu model, bir yatırım tavsiyesi vermekten ziyade, yatırımcının karar destek mekanizması olarak tasarlanmıştır.
*   **Canlıya Alma:** Model, `Streamlit` kullanılarak interaktif bir web arayüzüne dönüştürülmüştür.
*   **İzleme (Monitoring):** Canlı ortamda modelin başarısı "Doğru Yön Tahmini Yüzdesi" metriği ile haftalık olarak takip edilmelidir.

## 5. Proje Yapısı
```text
BIST_PREDICTION/
├── data/               # Ham ve işlenmiş veriler
├── models/             # Eğitilmiş .json/.pkl modeller
├── notebooks/          # EDA ve Deneme not defterleri
├── src/                # Kaynak kodlar
│   ├── config.py       # Ayarlar
│   ├── features.py     # İndikatör hesaplamaları
│   └── model_train.py  # Eğitim scripti
├── app.py              # Streamlit arayüz kodu
├── requirements.txt    # Kütüphane bağımlılıkları
└── README.md           # Proje dokümantasyonu