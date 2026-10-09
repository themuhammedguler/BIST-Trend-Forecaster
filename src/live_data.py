# src/live_data.py - Yahoo Finance'ten canlı veri çekip modele hazırlayan ortak yardımcılar
# (Tek hisse analizi ve BIST 30 taraması aynı işlem hattını kullanır.)
import pandas as pd

from src import config, features, network

# İndikatörlerin ısınması (sma_50) sonrası 6 aylık backtest için son 1 yılın verisi çekilir
LIVE_PERIOD = "1y"


def yahoo_symbol(ticker):
    """Yahoo Finance sembolü değişen hisseleri eşler (Örn: KOZAL -> TRALT)."""
    return config.TICKER_YAHOO_MAP.get(ticker, ticker)


def extract_symbol_frame(batch, symbol):
    """Toplu yf.download çıktısından tek bir sembolün OHLCV verisini çıkarır.

    Yahoo verisi alınamayan sembolleri tamamen NaN bir blok olarak döndürür;
    bu satırlar atılır, sembol hiç yoksa boş DataFrame döner.
    """
    if batch is None or batch.empty:
        return pd.DataFrame()
    if not isinstance(batch.columns, pd.MultiIndex):
        return batch
    for level in range(batch.columns.nlevels):
        if symbol in batch.columns.get_level_values(level):
            return batch.xs(symbol, axis=1, level=level).dropna(how="all")
    return pd.DataFrame()


def prepare_live_frame(raw, ticker, macro_data=None):
    """yf.download çıktısını features.add_features'ın beklediği biçime getirip
    indikatörleri ve makro piyasa göstergelerini hesaplar.

    Dönüş: (df_processed, df_ohlcv). df_processed'in son satırı en güncel işlem
    günüdür; df_ohlcv grafik ve fiyat metrikleri içindir.
    Veri yoksa veya indikatörler için yetersizse ValueError fırlatır.
    """
    if raw is None or raw.empty:
        raise ValueError(
            f"'{ticker}' (Yahoo: '{yahoo_symbol(ticker)}') için piyasa verisi alınamadı. "
            "Sembol değişmiş veya Yahoo Finance servisine geçici olarak ulaşılamıyor olabilir; "
            "lütfen birkaç dakika sonra tekrar deneyin."
        )

    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df["ticker"] = ticker.replace(".IS", "").replace("=X", "")
    df.reset_index(inplace=True)

    # Sütun isimlerini düzenle (features.py 'Date' ve küçük harfli sütunlar bekliyor)
    new_columns = {}
    for col in df.columns:
        if col.lower() in ("date", "index"):
            new_columns[col] = "Date"
        elif col.lower() == "ticker":
            new_columns[col] = "ticker"
        else:
            new_columns[col] = col.lower()
    df.rename(columns=new_columns, inplace=True)

    if "Date" not in df.columns:
        raise ValueError(f"'{ticker}' için çekilen veride 'Date' sütunu bulunamadı.")

    # Yahoo seans sonrası bazen günün satırını OHLC'si boş, yalnızca hacmi dolu
    # döndürür (#25). Bu satır fiyat, grafik ve tahminin farklı günleri
    # kullanmasına yol açtığı için atılır; hepsi son tam işlem gününü kullanır.
    price_cols = [c for c in ("open", "high", "low", "close") if c in df.columns]
    df = df.dropna(subset=price_cols).reset_index(drop=True)

    # drop_incomplete_target=False: canlı tahminde bugünün hedefi (yarının kapanışı)
    # henüz bilinmez; bu normalde eğitimde düşürülen son günü burada tutar.
    df_processed = features.add_features(df, macro_df=macro_data, drop_incomplete_target=False)
    if df_processed.empty:
        raise ValueError(
            f"'{ticker}' verisi teknik indikatörler hesaplandıktan sonra yetersiz kaldı."
        )

    return df_processed, df


def fetch_macro_frame(period=LIVE_PERIOD):
    """XU100.IS ve USDTRY=X için canlı verileri tek bir istekte indirir."""
    macro_tickers = getattr(
        config,
        "MACRO_TICKERS",
        [getattr(config, "INDEX_TICKER", "XU100.IS"), getattr(config, "FX_TICKER", "USDTRY=X")],
    )
    if not macro_tickers:
        return None
    try:
        return network.download_with_retry(
            macro_tickers,
            period=period,
            group_by="ticker",
            progress=False,
        )
    except Exception:
        return None


def fetch_live_frame(ticker, macro_data=None):
    """Tek bir hisse için canlı veriyi indirip prepare_live_frame ile hazırlar."""
    raw = network.download_with_retry(yahoo_symbol(ticker), period=LIVE_PERIOD, progress=False)
    return prepare_live_frame(raw, ticker, macro_data=macro_data)
