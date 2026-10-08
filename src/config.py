# src/config.py
import os

# Proje ana dizini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bist30_combined.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_bist_model.json")
MODEL_META_PATH = os.path.join(BASE_DIR, "models", "xgb_bist_model_meta.json")

# BIST 30 Hisseleri (Likiditesi yüksek, manipülasyonu zor)
TICKERS = [
    "AKBNK.IS",
    "ARCLK.IS",
    "ASELS.IS",
    "BIMAS.IS",
    "EKGYO.IS",
    "ENKAI.IS",
    "EREGL.IS",
    "FROTO.IS",
    "GARAN.IS",
    "GUBRF.IS",
    "HALKB.IS",
    "HEKTS.IS",
    "ISCTR.IS",
    "KCHOL.IS",
    "KOZAA.IS",
    "KOZAL.IS",
    "KRDMD.IS",
    "PETKM.IS",
    "PGSUS.IS",
    "SAHOL.IS",
    "SASA.IS",
    "SISE.IS",
    "TCELL.IS",
    "THYAO.IS",
    "TKFEN.IS",
    "TOASO.IS",
    "TSKB.IS",
    "TTKOM.IS",
    "TUPRS.IS",
    "YKBNK.IS",
]

# Yahoo Finance'te sembolü/adı değişen hisseler için eşleme tablosu
# (24 Kasım 2025: KOZAL -> TRALT (Türk Altın), KOZAA -> TRMET (TR Anadolu Metal))
TICKER_YAHOO_MAP = {
    "KOZAL.IS": "TRALT.IS",
    "KOZAA.IS": "TRMET.IS",
    "TRALT.IS": "TRALT.IS",
    "TRMET.IS": "TRMET.IS",
}

# Eğitim için kaç yıllık veri çekilsin?
START_DATE = "2018-01-01"
END_DATE = "2025-12-08"  # Bugüne kadar

# Yön sinyali için nötr bant: bu aralıktaki olasılıklar yazı-turadan ayırt edilemez
# prob >= HIGH -> yükseliş, prob <= LOW -> düşüş, arası -> nötr / belirsiz
PROB_THRESHOLD_HIGH = 0.53
PROB_THRESHOLD_LOW = 0.47

# Modelin girdi olarak kullandığı teknik göstergeler ve takvim öznitelikleri
# (Single Source of Truth)
FEATURES = [
    "rsi",
    "macd",
    "sma_10",
    "sma_50",
    "bb_width",
    "volatility",
    "lag_1_ret",
    "lag_2_ret",
    "vol_change",
    "day_of_week",
    "month",
]
