# src/model_train.py
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import config
import features
import joblib
import os

def train_model():
    # 1. Veriyi Yükle
    print("Veri yükleniyor...")
    df = pd.read_csv(config.DATA_PATH)
    
    # 2. Feature Engineering
    print("Feature Engineering uygulanıyor...")
    df_processed = features.add_features(df)
    
    # 3. Eğitim Seti Hazırlığı
    # Geleceği görmeyi engellemek için tarihsel kesim yapıyoruz (TimeSeries Split mantığı)
    features_list = ['rsi', 'macd', 'sma_10', 'sma_50', 'bb_width', 
                     'volatility', 'lag_1_ret', 'lag_2_ret', 'vol_change', 
                     'day_of_week', 'month']
    
    X = df_processed[features_list]
    y = df_processed['target']
    
    # Son 3 ayı test verisi olarak ayıralım, gerisi eğitim
    split_point = int(len(df_processed) * 0.9)
    X_train, X_test = X.iloc[:split_point], X.iloc[split_point:]
    y_train, y_test = y.iloc[:split_point], y.iloc[split_point:]
    
    print(f"Eğitim Verisi: {X_train.shape}, Test Verisi: {X_test.shape}")
    
    # 4. Model Tanımlama ve Eğitim (XGBoost)
    # PDF'teki 'Model Optimization' burada manuel parametrelerle simüle edildi
    model = xgb.XGBClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        objective='binary:logistic',
        random_state=42,
        n_jobs=-1
    )
    
    print("Model eğitiliyor...")
    model.fit(X_train, y_train)
    
    # 5. Değerlendirme
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n🎯 Model Doğruluğu (Test Seti): {acc:.4f}")
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, preds))
    
    # Feature Importance (PDF Maddesi: Model Evaluation)
    importance = dict(zip(features_list, model.feature_importances_))
    print("\nÖnem Düzeyleri:")
    for k, v in sorted(importance.items(), key=lambda item: item[1], reverse=True):
        print(f"{k}: {v:.4f}")
    
    # 6. Modeli Kaydet
    if not os.path.exists(os.path.dirname(config.MODEL_PATH)):
        os.makedirs(os.path.dirname(config.MODEL_PATH))
        
    model.save_model(config.MODEL_PATH)
    print(f"\n✅ Model kaydedildi: {config.MODEL_PATH}")

if __name__ == "__main__":
    train_model()