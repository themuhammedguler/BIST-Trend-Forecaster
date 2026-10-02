# src/scanner.py
# BIST 30 hisselerinin tamamını modelden geçirip fırsat radarı sıralaması oluşturan modül
import pandas as pd
import numpy as np
import yfinance as yf
from src import config, features


def predict_single_ticker(ticker, model, period="6mo"):
    """
    Tek bir hisse senedi için canlı piyasa verisini çeker, teknik indikatörleri hesaplar
    ve modelin artış olasılığı ile son fiyat metriklerini sözlük olarak döndürür.
    """
    yahoo_ticker = getattr(config, "TICKER_YAHOO_MAP", {}).get(ticker, ticker)
    df = yf.download(yahoo_ticker, period=period, progress=False)

    if df is None or df.empty or len(df) == 0:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["ticker"] = ticker.replace(".IS", "")
    df.reset_index(inplace=True)

    new_columns = {}
    for col in df.columns:
        if col.lower() in ("date", "index"):
            new_columns[col] = "Date"
        elif col.lower() == "ticker":
            new_columns[col] = "ticker"
        else:
            new_columns[col] = col.lower()

    df.rename(columns=new_columns, inplace=True)
    if "Date" not in df.columns or len(df) < 15:
        return None

    df_processed = features.add_features(df, drop_incomplete_target=False)
    if df_processed.empty:
        return None

    features_list = [
        "rsi", "macd", "sma_10", "sma_50", "bb_width",
        "volatility", "lag_1_ret", "lag_2_ret", "vol_change",
        "day_of_week", "month"
    ]
    last_row = df_processed.iloc[[-1]]
    X_pred = last_row[features_list]

    prob = float(model.predict_proba(X_pred)[0][1])
    curr_price = float(df["close"].iloc[-1]) if len(df) > 0 else 0.0
    change_rate = 0.0
    if len(df) >= 2:
        prev_price = float(df["close"].iloc[-2])
        if prev_price > 0:
            change_rate = float(((curr_price - prev_price) / prev_price) * 100)

    clean_name = ticker.replace(".IS", "")
    if clean_name == "KOZAL":
        clean_name = "KOZAL (TRALT)"
    elif clean_name == "KOZAA":
        clean_name = "KOZAA (TRMET)"

    return {
        "Hisse": clean_name,
        "ticker_code": ticker,
        "Son Fiyat (TL)": round(curr_price, 2),
        "Günlük Değişim (%)": round(change_rate, 2),
        "Yükseliş Olasılığı (%)": round(prob * 100, 1),
        "Tahmin": "YUKARI 🚀" if prob > 0.5 else "DÜŞÜŞ 🔻",
        "_prob": prob,
    }


def scan_market(tickers, model):
    """
    Verilen hisse listesini tarar, modelden geçirir ve artış olasılığına göre
    azalan sırada sıralanmış bir DataFrame döndürür.
    """
    records = []
    for ticker in tickers:
        try:
            res = predict_single_ticker(ticker, model)
            if res is not None:
                records.append(res)
        except Exception:
            continue

    if not records:
        return pd.DataFrame(
            columns=[
                "Hisse", "ticker_code", "Son Fiyat (TL)",
                "Günlük Değişim (%)", "Yükseliş Olasılığı (%)", "Tahmin", "_prob"
            ]
        )

    df_scan = pd.DataFrame(records)
    df_scan.sort_values(by="_prob", ascending=False, inplace=True)
    df_scan.reset_index(drop=True, inplace=True)
    return df_scan
