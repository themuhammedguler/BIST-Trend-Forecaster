# src/features.py
import numpy as np  # type: ignore
import pandas as pd  # type: ignore
import ta  # type: ignore  # Technical Analysis Library


def extract_macro_table(macro_df):
    """Verilen DataFrame'den (MultiIndex veya satır bazlı) XU100 ve USD/TRY
    günlük getiri serilerini çıkarıp Date indeksli bir tablo döndürür."""
    if macro_df is None or macro_df.empty:
        return None

    xu_ret = None
    fx_ret = None

    if isinstance(macro_df.columns, pd.MultiIndex):
        # Toplu yfinance indirme çıktısı (MultiIndex)
        for sym in ["XU100.IS", "XU100"]:
            for lvl in range(macro_df.columns.nlevels):
                if sym in macro_df.columns.get_level_values(lvl):
                    s = macro_df.xs(sym, axis=1, level=lvl)
                    col = (
                        "Close"
                        if "Close" in s.columns
                        else ("close" if "close" in s.columns else None)
                    )
                    if col:
                        xu_ret = s[col].pct_change()
                    break
            if xu_ret is not None:
                break

        for sym in ["USDTRY=X", "USDTRY"]:
            for lvl in range(macro_df.columns.nlevels):
                if sym in macro_df.columns.get_level_values(lvl):
                    s = macro_df.xs(sym, axis=1, level=lvl)
                    col = (
                        "Close"
                        if "Close" in s.columns
                        else ("close" if "close" in s.columns else None)
                    )
                    if col:
                        fx_ret = s[col].pct_change()
                    break
            if fx_ret is not None:
                break
    else:
        # Düz (flat) DataFrame: ticker ve Date sütunları
        if "ticker" in macro_df.columns and "Date" in macro_df.columns:
            xu_mask = macro_df["ticker"].isin(["XU100", "XU100.IS"])
            if xu_mask.any():
                sub = macro_df[xu_mask].sort_values("Date")
                col = (
                    "close"
                    if "close" in sub.columns
                    else ("Close" if "Close" in sub.columns else None)
                )
                if col:
                    xu_ret = pd.Series(
                        sub[col].values, index=pd.to_datetime(sub["Date"])
                    ).pct_change()

            fx_mask = macro_df["ticker"].isin(["USDTRY", "USDTRY=X"])
            if fx_mask.any():
                sub = macro_df[fx_mask].sort_values("Date")
                col = (
                    "close"
                    if "close" in sub.columns
                    else ("Close" if "Close" in sub.columns else None)
                )
                if col:
                    fx_ret = pd.Series(
                        sub[col].values, index=pd.to_datetime(sub["Date"])
                    ).pct_change()

    if xu_ret is None and fx_ret is None:
        return None

    ref_index = xu_ret.index if xu_ret is not None else fx_ret.index
    df_out = pd.DataFrame(index=pd.to_datetime(ref_index))
    if xu_ret is not None:
        xu_ret.index = pd.to_datetime(xu_ret.index)
        df_out["xu100_ret"] = xu_ret
    else:
        df_out["xu100_ret"] = 0.0

    if fx_ret is not None:
        fx_ret.index = pd.to_datetime(fx_ret.index)
        df_out["usdtry_change"] = fx_ret
    else:
        df_out["usdtry_change"] = 0.0

    return df_out.sort_index().ffill().fillna(0.0)


def add_features(df, macro_df=None, drop_incomplete_target=True):
    """Verilen DataFrame'e teknik analiz indikatörleri, makro piyasa göstergeleri
    ve zaman özellikleri ekler.

    PDF Gereksinimi: En az 10 feature.
    Makro Göstergeler (#10):
      - xu100_ret: BIST 100 endeksi günlük getirisi
      - rel_strength_bist: Hissenin BIST 100'e göre rölatif getirisi (hisse - xu100)
      - usdtry_change: USD/TRY kuru günlük değişimi

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

    # Makro veriyi ayıkla: df içinde 'XU100' veya 'USDTRY' satırları varsa ayır
    macro_table = None
    if macro_df is not None:
        macro_table = extract_macro_table(macro_df)
    elif "ticker" in df.columns:
        macro_tickers = {"XU100", "USDTRY", "XU100.IS", "USDTRY=X"}
        if df["ticker"].isin(macro_tickers).any():
            macro_rows = df[df["ticker"].isin(macro_tickers)].copy()
            df = df[~df["ticker"].isin(macro_tickers)].copy()
            if df.empty:
                return df.copy()
            macro_table = extract_macro_table(macro_rows)

    # df'e makro göstergeleri Date üzerinden birleştir
    if macro_table is not None and not macro_table.empty:
        df = df.merge(
            macro_table[["xu100_ret", "usdtry_change"]],
            left_on="Date",
            right_index=True,
            how="left",
        )
        df["_xu100_ret"] = df["xu100_ret"].ffill().fillna(0.0)
        df["_usdtry_change"] = df["usdtry_change"].ffill().fillna(0.0)
        df.drop(columns=["xu100_ret", "usdtry_change"], inplace=True)
    else:
        df["_xu100_ret"] = 0.0
        df["_usdtry_change"] = 0.0

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
    df.drop(columns=["_xu100_ret", "_usdtry_change"], inplace=True)
    df = pd.concat([df, new_cols], axis=1)

    # --- KRİTİK DÜZELTME ---
    # 1. Sonsuz değerleri (inf, -inf) NaN (boş) değere çevir
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # 2. NaN olan satırları sil (İndikatörlerin hesaplanamadığı ilk günler veya hatalı veriler)
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

    # 9. Makro Piyasa Göstergeleri (#10)
    out["xu100_ret"] = df["_xu100_ret"]
    out["rel_strength_bist"] = out["pct_change"] - df["_xu100_ret"]
    out["usdtry_change"] = df["_usdtry_change"]

    # Target
    out["next_close"] = close.shift(-1)
    out["target"] = (out["next_close"] > close).astype(int)
    return pd.DataFrame(out, index=df.index)
