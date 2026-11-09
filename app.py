# app.py
import os

import streamlit as st

from src import config, model_metadata, signals
from src.ui import (
    get_cached_macro_data,
    get_cached_market_scan,
    get_prediction_data,
    load_model,
    render_market_scanner_tab,
    render_safely,
    render_sidebar_model_info,
    render_single_ticker_tab,
)

# Geriye dönük uyumluluk için dışa aktarımlar
__all__ = [
    "get_cached_macro_data",
    "get_cached_market_scan",
    "get_prediction_data",
    "load_model",
]

# Sayfa Ayarları & Özel Favicon / Logo
LOGO_PATH = os.path.join(os.path.dirname(__file__), "assets", "logo.png")
PAGE_ICON = LOGO_PATH if os.path.exists(LOGO_PATH) else "📈"

st.set_page_config(
    page_title="BIST Trend Forecaster | Yapay Zeka Yön Tahmini",
    page_icon=PAGE_ICON,
    layout="wide",
)

st.title("📈 Borsa İstanbul Yapay Zeka Yön Tahmini", anchor=False)
st.markdown("""
Bu proje **XGBoost** algoritması kullanarak BIST 30 hisselerinin
bir sonraki günkü kapanış yönünü (Artış/Düşüş) tahmin eder.
""")

# Yan Menü: Logo ve Hisse Seçimi & Deep-Linking Desteği
if os.path.exists(LOGO_PATH):
    st.sidebar.image(LOGO_PATH, width=80)
st.sidebar.header("Hisse Seçimi")
ticker_list = [t.replace(".IS", "") for t in config.TICKERS]

query_params = st.query_params
raw_param = query_params.get("ticker") or query_params.get("symbol")
default_idx = 0
if raw_param:
    clean_param = str(raw_param).upper().strip().replace(".IS", "")
    if clean_param in ticker_list:
        default_idx = ticker_list.index(clean_param)

selected_ticker = st.sidebar.selectbox(
    "Hisse Senedi Seçiniz",
    ticker_list,
    index=default_idx,
    format_func=lambda x: (
        f"{x} (TRALT)" if x == "KOZAL" else (f"{x} (TRMET)" if x == "KOZAA" else x)
    ),
)
# Tarayıcı URL'sini seçili hisseyle senkronize et (paylaşılabilir derin bağlantı)
st.query_params["ticker"] = selected_ticker
selected_ticker_full = selected_ticker + ".IS"

# Veriler 15 dk önbellekte tutulur; buton yalnızca seçili hissenin kaydını temizler (#11)
st.sidebar.button(
    "🔄 Verileri Yenile",
    key="btn_refresh_data",
    help="Seçili hissenin verisini önbelleği atlayarak yeniden indirir.",
    on_click=lambda: get_prediction_data.clear(selected_ticker_full),
)

# Yan Menü: Sinyal & Strateji Ayarları
st.sidebar.header("Sinyal & Strateji Ayarları")
min_confidence = st.sidebar.slider(
    "Minimum Güven Eşiği",
    min_value=0.51,
    max_value=0.70,
    value=config.PROB_THRESHOLD_HIGH,
    step=0.01,
    help="Olasılık bu eşiğin üzerinde (ya da 1 - eşik altında) değilse sinyal nötr gösterilir.",
)
prob_low, prob_high = signals.symmetric_band(min_confidence)

commission_rate = st.sidebar.slider(
    "Komisyon & Kayma Oranı (%)",
    min_value=0.0,
    max_value=0.50,
    value=0.0,
    help=(
        "Pozisyon değişimlerinde (alım veya satım) düşülecek işlem maliyeti "
        "(BIST ortalaması ~%0.10)."
    ),
)
cost = commission_rate / 100.0

# Yan Menü: Model Bilgisi (#13, #45)
model_meta = model_metadata.load_metadata(config.MODEL_META_PATH)
render_sidebar_model_info(model_meta)

# Ana Akış
if not os.path.exists(config.MODEL_PATH):
    st.error("Model dosyası bulunamadı! Lütfen önce `src/model_train.py` çalıştırın.")
else:
    tab1, tab2 = st.tabs(["🎯 Tek Hisse Analizi", "📊 BIST 30 Fırsat Radarı"])

    with tab1:
        render_safely(
            lambda: render_single_ticker_tab(
                selected_ticker, selected_ticker_full, prob_low, prob_high, cost
            )
        )

    with tab2:
        render_safely(render_market_scanner_tab)
