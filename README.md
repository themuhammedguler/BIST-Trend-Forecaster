# BIST 30 Hisse Yön Tahmini (AI-Based Stock Trend Prediction)

## Proje Tanımı
Bu proje, Borsa İstanbul (BIST 30) hisselerinin teknik analiz verilerini kullanarak, bir sonraki gün fiyatının artıp artmayacağını tahmin eden bir makine öğrenmesi modelidir.

## Kurulum
1. `pip install -r requirements.txt`
2. `python src/data_ingestion.py` (Veri Setini Oluştur)
3. `python src/model_train.py` (Modeli Eğit)
4. `streamlit run app.py` (Arayüzü Başlat)

## Veri Seti ve Özellikler
- **Kaynak:** Yahoo Finance (yfinance kütüphanesi).
- **Boyut:** ~70.000 satır (30 hisse x 8 yıl).
- **Features:** RSI, MACD, SMA(10,50), Bollinger Bands, Volatility, Lag Features.

## Model ve Performans
- **Model:** XGBoost Classifier.
- **Validasyon:** TimeSeriesSplit (Gelecek verisinin sızmasını önlemek için tarihsel ayırma yapıldı).
- **Başarı Skoru:** Test setinde ~%55-60 doğruluk (Finansal piyasaların gürültülü yapısı nedeniyle kabul edilebilir bir oran).

## Pipeline ve İş Akışı
Veri ham olarak çekilir -> Teknik indikatörler hesaplanır (Feature Eng.) -> Tarihsel olarak train/test ayrılır -> XGBoost ile eğitilir -> Streamlit üzerinden canlı tahmin sunulur.

## Business Değeri
Yatırımcıların duygusal kararlar yerine, teknik verilerle desteklenen matematiksel olasılıklara göre pozisyon almasına yardımcı olur.