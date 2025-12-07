# src/features.py
import pandas as pd # type: ignore
import numpy as np # type: ignore
import ta # type: ignore # Technical Analysis Library

def add_features(df):
    """
    Verilen DataFrame'e teknik analiz indikatörleri ve zaman özellikleri ekler.
    PDF Gereksinimi: En az 10 feature.
    """
    df = df.copy()
    
    # Datetime dönüşümü
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values(by=['ticker', 'Date'])
    
    # Her hisse için ayrı hesaplama yapılmalı (Groupby)
    # 1. RSI
    df['rsi'] = df.groupby('ticker')['close'].transform(lambda x: ta.momentum.rsi(x, window=14))
    
    # 2. MACD
    df['macd'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.macd_diff(x))
    
    # 3. Hareketli Ortalamalar
    df['sma_10'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.sma_indicator(x, window=10))
    df['sma_50'] = df.groupby('ticker')['close'].transform(lambda x: ta.trend.sma_indicator(x, window=50))
    
    # 4. Bollinger Bands
    df['bb_width'] = df.groupby('ticker')['close'].transform(lambda x: ta.volatility.bollinger_wband(x))
    
    # 5. Volatilite
    df['volatility'] = df.groupby('ticker')['close'].transform(lambda x: x.rolling(10).std())
    
    # 6. Lag Features
    # pct_change bazen 0'a bölünme yüzünden inf üretebilir, bunu aşağıda temizleyeceğiz
    df['pct_change'] = df.groupby('ticker')['close'].pct_change()
    df['lag_1_ret'] = df.groupby('ticker')['pct_change'].shift(1)
    df['lag_2_ret'] = df.groupby('ticker')['pct_change'].shift(2)
    
    # 7. Tarihsel Özellikler
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['month'] = df['Date'].dt.month
    
    # 8. Hacim Değişimi
    df['vol_change'] = df.groupby('ticker')['volume'].pct_change()
    
    # Target
    df['next_close'] = df.groupby('ticker')['close'].shift(-1)
    df['target'] = (df['next_close'] > df['close']).astype(int)
    
    # --- KRİTİK DÜZELTME ---
    # 1. Sonsuz değerleri (inf, -inf) NaN (boş) değere çevir
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 2. NaN olan satırları sil (İndikatörlerin hesaplanamadığı ilk günler veya hatalı veriler)
    df.dropna(inplace=True)
    # -----------------------
    
    return df