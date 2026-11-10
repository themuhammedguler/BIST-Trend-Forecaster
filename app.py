# app.py
import base64
import os

import streamlit as st

from src import analytics, config, signals
from src.ui import (
    get_cached_macro_data,
    get_cached_market_scan,
    get_prediction_data,
    load_model,
    render_market_scanner_tab,
    render_safely,
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
    /* Sinyal & Strateji Ayarları Slider Renk Özelleştirmesi (Açık Yeşil / Zümrüt) */
    div[data-testid="stSlider"] div[data-testid="stSliderThumbValue"],
    div[data-baseweb="slider"] [data-testid="stSliderThumbValue"],
    div[data-baseweb="slider"] div[role="slider"] div {
        color: #34D399 !important;
        font-weight: 700 !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"] {
        background-color: #10B981 !important;
        border-color: #34D399 !important;
        box-shadow: 0 0 10px rgba(16, 185, 129, 0.45) !important;
    }

    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]:focus,
    div[data-testid="stSlider"] [data-baseweb="slider"] div[role="slider"]:hover {
        background-color: #34D399 !important;
        box-shadow: 0 0 14px rgba(52, 211, 153, 0.65) !important;
    }

    /* Slider Dolgulu İlerleme Çubuğu (Active Track) */
    div[data-testid="stSlider"] [data-baseweb="slider"] > div > div:first-child {
        background-color: #10B981 !important;
    }

    /* Tooltip / Açılan Balon */
    div[data-baseweb="tooltip"] {
        background-color: #0F131D !important;
        color: #34D399 !important;
        border: 1px solid #10B981 !important;
    }

    /* ==================================================================== */
    /* ANALİZ MENÜSÜ: Modern Dikey Navigasyon Menüsü (Radyo Noktalarını Gizle) */
    /* ==================================================================== */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 6px !important;
        display: flex !important;
        flex-direction: column !important;
    }

    /* Radyo dairesini ve radyo noktasını tamamen gizle */
    div[data-testid="stSidebar"] div[data-testid="stRadio"]
      label[data-baseweb="radio"] > div:first-child,
    div[data-testid="stSidebar"] div[data-testid="stRadio"] input[type="radio"] {
        display: none !important;
        visibility: hidden !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    /* Menü kartının şık, modern buton tasarımı */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"] {
        background: linear-gradient(
            180deg,
            rgba(30, 41, 59, 0.70) 0%,
            rgba(15, 23, 42, 0.85) 100%
        ) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 8px !important;
        padding: 9px 12px !important;
        margin: 0 !important;
        cursor: pointer !important;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
        display: flex !important;
        align-items: center !important;
        width: 100% !important;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.2) !important;
    }

    /* Hover: Modern neon mavi parlama ve hafif kayma efekti */
    div[data-testid="stSidebar"] div[data-testid="stRadio"] label[data-baseweb="radio"]:hover {
        background: linear-gradient(
            180deg,
            rgba(37, 99, 235, 0.18) 0%,
            rgba(30, 41, 59, 0.90) 100%
        ) !important;
        border-color: rgba(96, 165, 250, 0.55) !important;
        transform: translateX(4px) !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.25) !important;
    }

    /* Aktif / Seçili menü kartı: Parlayan mavi gradyan ve sol vurgu çizgisi */
    div[data-testid="stSidebar"] div[data-testid="stRadio"]
      label[data-baseweb="radio"]:has(input:checked),
    div[data-testid="stSidebar"] div[data-testid="stRadio"]
      label[data-baseweb="radio"][aria-checked="true"] {
        background: linear-gradient(
            135deg,
            rgba(37, 99, 235, 0.35) 0%,
            rgba(15, 23, 42, 0.95) 100%
        ) !important;
        border: 1px solid #3b82f6 !important;
        border-left: 4px solid #60a5fa !important;
        box-shadow: 0 0 12px rgba(59, 130, 246, 0.32) !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"]
      label[data-baseweb="radio"] div[data-testid="stMarkdownContainer"] p {
        font-size: 0.88rem !important;
        font-weight: 500 !important;
        color: #e2e8f0 !important;
        margin: 0 !important;
        line-height: 1.35 !important;
    }

    div[data-testid="stSidebar"] div[data-testid="stRadio"]
      label[data-baseweb="radio"]:has(input:checked)
      div[data-testid="stMarkdownContainer"] p {
        color: #ffffff !important;
        font-weight: 600 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

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

# Yan Menü: Analiz Modülü Seçimi
st.sidebar.header("Analiz Menüsü")
VIEW_OPTIONS = [
    "📈 Teknik Model & Grafikler",
    "🏢 Kurumsal Bilanço & Chatbot",
    "📑 KAP Bildirimleri & Analiz",
    "💼 Paper Trading Simülatörü",
    "🎯 Model Doğrulama (Audit)",
    "🔬 Veri Sapması & Kararlılık (PSI)",
]
selected_view = st.sidebar.radio(
    "İnceleme Modülü",
    VIEW_OPTIONS,
    index=0,
    key="sidebar_analysis_view",
    label_visibility="collapsed",
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

# Yan Menü: Analitik Durumu
analytics.render_analytics_status()


def _render_single_ticker_tab_safe():
    try:
        render_single_ticker_tab(
            selected_ticker,
            selected_ticker_full,
            prob_low,
            prob_high,
            cost,
            active_view=selected_view,
        )
    except TypeError as te:
        if "active_view" in str(te):
            # Sıcak yeniden yükleme (hot reload) esnasında modül önbelleği uyumluluğu
            render_single_ticker_tab(
                selected_ticker,
                selected_ticker_full,
                prob_low,
                prob_high,
                cost,
            )
        else:
            raise te


is_main_technical_view = selected_view == VIEW_OPTIONS[0]

# Ana Sayfa Başlığı, Açıklama ve Sekmeler yalnızca ana teknik görünümde gösterilir
if is_main_technical_view:
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

# Ana Akış
if not os.path.exists(config.MODEL_PATH):
    st.error("Model dosyası bulunamadı! Lütfen önce `src/model_train.py` çalıştırın.")
elif is_main_technical_view:
    tab1, tab2 = st.tabs(["🎯 Tek Hisse Analizi", "📊 BIST 30 Fırsat Radarı"])

    with tab1:
        render_safely(_render_single_ticker_tab_safe)

    with tab2:
        render_safely(render_market_scanner_tab)
else:
    render_safely(_render_single_ticker_tab_safe)
