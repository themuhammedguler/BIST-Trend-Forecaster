# app.py
import base64
import os

import streamlit as st

from src import analytics, config, model_metadata, signals
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
FAVICON_PATH = os.path.join(os.path.dirname(__file__), "assets", "favicon.png")
PAGE_ICON = (
    FAVICON_PATH
    if os.path.exists(FAVICON_PATH)
    else (LOGO_PATH if os.path.exists(LOGO_PATH) else "📈")
)

st.set_page_config(
    page_title="BIST Trend Forecaster | Yapay Zeka Yön Tahmini",
    page_icon=PAGE_ICON,
    layout="wide",
)

# Ziyaretçi ve Oturum Takibi (GoatCounter / Dahili Sayaç)
analytics.record_session()
analytics.inject_tracking_script()

# Logo Base64 Çevrimi (HTML olarak render edilip Streamlit fullscreen butonu engellenir)
def get_logo_base64() -> str | None:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")
    return None


LOGO_B64 = get_logo_base64()

# Özel Stil İyileştirmeleri (Fullscreen butonu ve lightbox engelleme)
st.markdown(
    """
    <style>
    /* Streamlit element toolbar ve fullscreen / enlarge butonlarını kesin olarak gizle */
    [data-testid="stElementToolbar"],
    [data-testid="stElementToolbarButton"],
    [data-testid="StyledFullScreenButton"],
    button[title*="fullscreen" i],
    button[aria-label*="fullscreen" i],
    button[title*="enlarge" i],
    button[aria-label*="enlarge" i],
    [data-testid="stImage"] button,
    .stImage button {
        display: none !important;
        visibility: hidden !important;
        opacity: 0 !important;
        pointer-events: none !important;
    }
    .brand-logo-img {
        border-radius: 50% !important;
        box-shadow: 0 0 12px rgba(0, 229, 255, 0.22);
        pointer-events: none !important;
        cursor: default !important;
        user-select: none !important;
        -webkit-user-drag: none !important;
        display: block;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Sayfa Başlığı ve Kurumsal Logo (HTML img olarak render edilir, fullscreen butonu çıkmaz)
if LOGO_B64:
    c_icon, c_title = st.columns([1, 16], vertical_alignment="center")
    with c_icon:
        st.markdown(
            f'<img src="data:image/png;base64,{LOGO_B64}" width="52" height="52" '
            f'class="brand-logo-img" alt="BIST Forecaster Logo" />',
            unsafe_allow_html=True,
        )
    with c_title:
        st.title("Borsa İstanbul Yapay Zeka Yön Tahmini", anchor=False)
else:
    st.title("Borsa İstanbul Yapay Zeka Yön Tahmini", anchor=False)
st.markdown("""
Bu platform; **CatBoost, LightGBM ve XGBoost** topluluk modelleri ile BIST 30
hisselerinin **1 Günlük, 1 Haftalık ve 1 Aylık** yön eğilimlerini ve risk analizini sunar.
""")

# Yan Menü: Kurumsal Marka Başlığı ve Hisse Seçimi
if LOGO_B64:
    c_logo, c_brand = st.sidebar.columns([1, 3], vertical_alignment="center")
    with c_logo:
        st.markdown(
            f'<img src="data:image/png;base64,{LOGO_B64}" width="54" height="54" '
            f'class="brand-logo-img" alt="BIST Forecaster Logo" />',
            unsafe_allow_html=True,
        )
    with c_brand:
        st.markdown(
            "<div style='line-height: 1.25; padding-top: 2px;'>"
            "<strong style='color: #F8FAFC; font-size: 1.05rem;'>BIST Forecaster</strong><br/>"
            "<span style='color: #94A3B8; font-size: 0.74rem;'>AI Karar Destek Platformu</span>"
            "</div>",
            unsafe_allow_html=True,
        )
    st.sidebar.markdown(
        "<hr style='margin: 10px 0 16px 0; border: none; border-top: 1px solid #1A2233;'/>",
        unsafe_allow_html=True,
    )
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

# Yan Menü: Model Bilgisi (#13, #45) & Analitik Durumu
model_meta = model_metadata.load_metadata(config.MODEL_META_PATH)
render_sidebar_model_info(model_meta)
analytics.render_analytics_status()

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
