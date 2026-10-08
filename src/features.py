# src/features.py
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import ta  # type: ignore  # Technical Analysis Library


def add_features(df, drop_incomplete_target=True):
    """Verilen DataFrame'e teknik analiz indikatörleri ve zaman özellikleri ekler.

    PDF Gereksinimi: En az 10 feature.

    drop_incomplete_target=True (eğitim): next_close/target hesaplanamayan
    (henüz bir sonraki günü bilinmeyen) en son gün her hisse için düşürülür.

    drop_incomplete_target=False (canlı tahmin): o son gün tutulur; 'target'
    sütunu bu satırlar için anlamsızdır ve kullanılmamalıdır, çünkü gerçek
    hedef henüz bilinmemektedir.
    """
    if df.empty:
        return df.copy()

    df = df.copy()

    # Datetime dönüşümü
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by=["ticker", "Date"])

    # Canlı tahmin veya tek hisse analizinde groupby genel giderini atlayıp
    # doğrudan seriler üzerinde vektörize hesaplama yapılır (#14)
    if df["ticker"].nunique() == 1:
        new_cols = _ticker_features(df)
    else:
        # Çoklu hisse senetleri için her hisse bağımsız olarak hesaplanır
        new_cols = pd.concat(
            _ticker_features(group) for _, group in df.groupby("ticker", sort=False)
        )
    df = pd.concat([df, new_cols], axis=1)

    # --- KRİTİK DÜZELTME ---
    # 1. Sonsuz değerleri (inf, -inf) NaN (boş) değere çevir
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. NaN olan satırları sil (İndikatörlerin hesaplanamadığı ilk günler veya hatalı veriler)
    # next_close/target, bir sonraki günü olmayan (henüz bilinmeyen) en son satırlar
    # için her zaman NaN/anlamsızdır; drop_incomplete_target=False ile canlı tahmin
    # senaryosunda bu satırlar sadece bu yüzden atılmaz.
    always_required = [c for c in df.columns if c not in ("next_close", "target")]
    df.dropna(subset=always_required, inplace=True)
    if drop_incomplete_target:
        df.dropna(subset=["next_close"], inplace=True)
    # -----------------------

    return df


def _ticker_features(df):
    """Tek bir hissenin tarihe göre sıralı verisinden indikatör ve hedef
    sütunlarını hesaplar; yalnızca yeni sütunları döndürür."""
    close = df["close"]
    out = {}

    # 1. RSI
    out["rsi"] = ta.momentum.rsi(close, window=14)

    # 2. MACD
    out["macd"] = ta.trend.macd_diff(close)

    # 3. Hareketli Ortalamalar
    out["sma_10"] = ta.trend.sma_indicator(close, window=10)
    out["sma_50"] = ta.trend.sma_indicator(close, window=50)

    # 4. Bollinger Bands
    out["bb_width"] = ta.volatility.bollinger_wband(close)

    # 5. Volatilite
    out["volatility"] = close.rolling(10).std()

    # 6. Lag Features
    # pct_change bazen 0'a bölünme yüzünden inf üretebilir, add_features bunu temizler
    out["pct_change"] = close.pct_change()
    out["lag_1_ret"] = out["pct_change"].shift(1)
    out["lag_2_ret"] = out["pct_change"].shift(2)

    # 7. Tarihsel Özellikler
    out["day_of_week"] = df["Date"].dt.dayofweek
    out["month"] = df["Date"].dt.month

    # 8. Hacim Değişimi
    out["vol_change"] = df["volume"].pct_change()

    # Target
    out["next_close"] = close.shift(-1)
    out["target"] = (out["next_close"] > close).astype(int)
    return pd.DataFrame(out, index=df.index)
