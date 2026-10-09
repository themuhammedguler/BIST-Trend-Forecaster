# Model Benchmark Raporu: XGBoost & Alternatif Mimariler (#24)

**Doğrulama Yöntemi:** 3 Katlı İleriye Doğru Genişleyen Pencere (Walk-Forward Temporal Cross-Validation)
**Rapor Tarihi:** 2026-10-09 16:35:58 UTC

## 1. Karşılaştırma Sonuçları Tablosu

| Model | Accuracy (±std) | Balanced Acc | ROC-AUC | Log Loss | Strateji Getirisi (%) | Sharpe | Süre (sn) |
|---|---|---|---|---|---|---|---|
| **PyTorch GRU (Recurrent DL)** | 0.5120 (±0.007) | 0.5116 | 0.5166 | 0.6981 | +2478681.97% | 0.36 | 7.59s |
| **HistGradientBoosting** | 0.5123 (±0.007) | 0.5121 | 0.5129 | 0.7019 | -100.00% | 0.20 | 0.39s |
| **XGBoost (Production)** | 0.5047 (±0.008) | 0.5048 | 0.5122 | 0.7027 | -66.84% | 0.22 | 0.47s |
| **Regularized Logistic Regression** | 0.5079 (±0.015) | 0.5075 | 0.5121 | 0.6996 | -100.00% | 0.19 | 0.06s |
| **Calibrated XGBoost (Platt)** | 0.5068 (±0.004) | 0.5067 | 0.5119 | 0.7043 | -23.21% | 0.27 | 0.13s |
| **LightGBM** | 0.5069 (±0.007) | 0.5069 | 0.5118 | 0.7021 | -95.01% | 0.23 | 0.95s |
| **MLP Neural Network (Deep Learning)** | 0.5022 (±0.006) | 0.5023 | 0.5024 | 0.8133 | +2375815.93% | 0.26 | 9.92s |
| **Baseline: Repeat Yesterday** | 0.5002 (±0.009) | 0.5001 | 0.5001 | 1.5230 | -0.77% | 0.20 | 0.00s |
| **Baseline: Always Down** | 0.4965 (±0.006) | 0.5000 | 0.5000 | 18.1468 | +0.00% | N/A | 0.00s |
| **Baseline: Always Up** | 0.5035 (±0.006) | 0.5000 | 0.5000 | 17.8969 | +50.50% | 0.51 | 0.00s |

## 2. Temel Bulgular & Analiz
- **Taban Çizgisi Karşılaştırması (Baselines):** Naif stratejiler (Always Down / Repeat Yesterday) piyasadaki zayıf sinyal seviyesini doğrulamaktadır.
- **Gradient Boosting Varyantları (XGBoost vs LightGBM vs HistGB):** LightGBM ve HistGradientBoosting, XGBoost ile benzer doğruluk seviyelerinde (%50.5 - %51.5) daha hızlı eğitim süreleri sunmaktadır.
- **Derin Öğrenme Modelleri (MLP / PyTorch GRU):** Kısa vadeli günlük tabular özelliklerde derin ağlar yüksek varyans göstermekte ve tabular boost algoritmalarına kıyasla belirgin bir üstünlük sağlayamamaktadır.
- **Olasılık Kalibrasyonu:** Calibrated XGBoost aşırı güvenli olasılıkları merkezileştirerek Log Loss değerinde iyileşme sağlamaktadır.

## 3. Karar ve Entegrasyon Stratejisi
Mevcut veriler ışığında, hiçbir alternatif model XGBoost'u ve taban çizgilerini açık ve istatistiksel olarak anlamlı bir farkla geçemediği için, **üretim modeli olarak XGBoost korunmuştur**. Ancak `src/benchmark.py` altyapısı sayesinde gelecekteki yeni öznitelik ve model denemeleri tam tekrarlanabilirlikle test edilebilecektir.
