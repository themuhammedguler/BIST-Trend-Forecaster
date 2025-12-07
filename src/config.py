# src/config.py
import os

# Proje ana dizini
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "bist30_combined.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "xgb_bist_model.json")

# BIST 30 Hisseleri (Likiditesi yüksek, manipülasyonu zor)
TICKERS = [
    "AKBNK.IS", "ARCLK.IS", "ASELS.IS", "BIMAS.IS", "EKGYO.IS", "ENKAI.IS", "EREGL.IS",
    "FROTO.IS", "GARAN.IS", "GUBRF.IS", "HALKB.IS", "HEKTS.IS", "ISCTR.IS", "KCHOL.IS",
    "KOZAA.IS", "KOZAL.IS", "KRDMD.IS", "PETKM.IS", "PGSUS.IS", "SAHOL.IS", "SASA.IS",
    "SISE.IS", "TCELL.IS", "THYAO.IS", "TKFEN.IS", "TOASO.IS", "TSKB.IS", "TTKOM.IS",
    "TUPRS.IS", "YKBNK.IS"
]

# Eğitim için kaç yıllık veri çekilsin?
START_DATE = "2018-01-01"
END_DATE = "2025-12-08" # Bugüne kadar