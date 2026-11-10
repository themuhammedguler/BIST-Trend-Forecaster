# src/scanner.py
# BIST 30 hisselerinin tamamını modelden geçirip fırsat radarı sıralaması oluşturan modül
import pandas as pd

from src import config, features, live_data, live_market, network


def predict_single_ticker(ticker, model):
    """
    Tek bir hisse senedi için canlı piyasa verisini çeker, teknik indikatörleri hesaplar
    ve modelin artış olasılığı ile son fiyat metriklerini sözlük olarak döndürür.
    """
    try:
        df_processed, df = live_data.fetch_live_frame(ticker)
    except ValueError:
        return None
    live_quotes = live_market.fetch_live_quotes([ticker])
    norm_sym = ticker.replace(".IS", "")
    live_q = live_quotes.get(norm_sym) if live_market.is_live_frame_recent(df) else None
    if live_q is not None:
        df_processed, df = live_market.apply_live_quote_to_frames(df_processed, df, live_q)
    return _build_record(ticker, df_processed, df, model, live_quote=live_q)


def _build_record(ticker, df_processed, df, model, live_quote=None):
    """Hazırlanmış canlı veriden tahmin ve fiyat metriklerini içeren satırı üretir."""
    last_row = df_processed.iloc[[-1]]
    X_pred = features.align_model_features(model, last_row)

    prob = float(model.predict_proba(X_pred)[0][1])
    if live_quote is not None and live_quote.last_price > 0:
        curr_price = float(live_quote.last_price)
        change_rate = float(live_quote.change_pct)
    else:
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

    vol = float(last_row["volatility"].iloc[0]) if "volatility" in last_row.columns else 0.25

    return {
        "Hisse": clean_name,
        "Sektör": config.get_ticker_sector(ticker),
        "ticker_code": ticker,
        "Son Fiyat (TL)": round(curr_price, 2),
        "Günlük Değişim (%)": round(change_rate, 2),
        "Yükseliş Olasılığı (%)": round(prob * 100, 1),
        "Tahmin": "YUKARI 🚀" if prob > 0.5 else "DÜŞÜŞ 🔻",
        "_prob": prob,
        "_volatility": vol,
        "volatility": round(vol, 4),
    }


def _symbol_frame(batch, symbol):
    """Toplu yf.download çıktısından tek bir sembolün OHLCV verisini çıkarır."""
    return live_data.extract_symbol_frame(batch, symbol)


def scan_market(tickers, model, include_macro=False):
    """
    Verilen hisse listesini tarar, modelden geçirir ve artış olasılığına göre
    azalan sırada sıralanmış bir DataFrame döndürür.

    Tüm semboller tek bir toplu Yahoo Finance isteğiyle indirilir.

    Dönüş: (df_scan, failed). failed, verisi alınamayan veya indikatörler için
    yetersiz kalan hisselerin listesidir. Veriyle ilgisi olmayan hatalar
    (örn. model/öznitelik uyumsuzluğu) gizlenmez, çağırana iletilir.
    """
    records = []
    failed = []
    symbols = {ticker: live_data.yahoo_symbol(ticker) for ticker in tickers}
    batch = None
    if symbols:
        try:
            download_symbols = list(symbols.values())
            if include_macro:
                macro_syms = getattr(config, "MACRO_TICKERS", [])
                download_symbols = download_symbols + [
                    s for s in macro_syms if s not in download_symbols
                ]
            batch = network.download_with_retry(
                sorted(set(download_symbols)),
                period=live_data.LIVE_PERIOD,
                group_by="ticker",
                progress=False,
            )
        except network.MarketDataUnavailableError:
            # Yahoo'ya ulaşılamadı: tüm hisseler verisi alınamayanlar listesine düşer
            batch = None
    live_quotes = live_market.fetch_live_quotes(list(tickers)) if symbols else {}
    for ticker, symbol in symbols.items():
        try:
            df_processed, df = live_data.prepare_live_frame(
                _symbol_frame(batch, symbol),
                ticker,
                macro_data=batch if include_macro else None,
            )
        except ValueError:
            failed.append(ticker)
            continue
        norm_sym = ticker.replace(".IS", "")
        live_q = live_quotes.get(norm_sym) if live_market.is_live_frame_recent(df) else None
        if live_q is not None:
            df_processed, df = live_market.apply_live_quote_to_frames(
                df_processed,
                df,
                live_q,
                macro_data=batch if include_macro else None,
            )
        # Model hataları (XGBoost öznitelik uyumsuzluğunda da ValueError fırlatır)
        # veri hatası sayılmamalı; bu yüzden try bloğunun dışında.
        records.append(_build_record(ticker, df_processed, df, model, live_quote=live_q))

    if not records:
        return pd.DataFrame(
            columns=[
                "Hisse",
                "Sektör",
                "ticker_code",
                "Son Fiyat (TL)",
                "Günlük Değişim (%)",
                "Yükseliş Olasılığı (%)",
                "Tahmin",
                "_prob",
            ]
        ), failed

    df_scan = pd.DataFrame(records)
    df_scan.sort_values(by="_prob", ascending=False, inplace=True)
    df_scan.reset_index(drop=True, inplace=True)
    return df_scan, failed


def top_and_bottom(df_scan, n=5):
    """Olasılığa göre sıralı taramadan en yüksek ve en düşük n hisseyi döndürür.

    Sonuç sayısı 2n'den azsa iki liste çakışmasın diye n küçültülür
    (örn. 9 hisse -> 4 + 4). Düşüş listesi en düşük olasılıktan başlar.
    """
    n = min(n, len(df_scan) // 2)
    if n == 0:
        return df_scan.iloc[0:0], df_scan.iloc[0:0]
    return df_scan.head(n), df_scan.tail(n).iloc[::-1]
