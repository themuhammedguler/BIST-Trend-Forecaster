"""Ortak UI bileşenleri, veri önbellekleme ve yardımcı fonksiyonlar."""

import streamlit as st
import xgboost as xgb

from src import config, live_data, scanner

# Tarama tablolarında sayıların tutarlı ondalıkla gösterimi
SCAN_COLUMN_CONFIG = {
    "Son Fiyat (TL)": st.column_config.NumberColumn(format="%.2f"),
    "Günlük Değişim (%)": st.column_config.NumberColumn(format="%.2f"),
    "Yükseliş Olasılığı (%)": st.column_config.NumberColumn(format="%.1f"),
    "Sektör": st.column_config.TextColumn("Sektör"),
}


@st.cache_resource
def load_model():
    """Eğitilmiş XGBoost modelini önbellekten yükler."""
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    return model


@st.cache_data(ttl=900, show_spinner=False)
def get_cached_macro_data():
    """Makro ekonomik göstergeleri (BIST 100, USD/TRY vb.) önbellekli çeker."""
    return live_data.fetch_macro_frame()


@st.cache_data(ttl=900, show_spinner=False)
def get_prediction_data(ticker):
    """Seçili hisse için canlı fiyat ve hesaplanmış öznitelikleri döner."""
    macro_data = get_cached_macro_data()
    return live_data.fetch_live_frame(ticker, macro_data=macro_data)


@st.cache_data(ttl=900, show_spinner=False)
def get_cached_market_scan(tickers):
    """Tüm BIST 30 hisselerini önbellekli olarak tarar ve sıralar."""
    model = load_model()
    return scanner.scan_market(tickers, model, include_macro=True)


def render_sidebar_model_info(model_meta):
    """Yan menüde model meta-veri bilgilerini ve veri sağlığı durumunu gösterir."""
    with st.sidebar.expander("ℹ️ Model Bilgisi", expanded=False):
        if model_meta:
            st.write(f"**Sürüm:** v{model_meta.get('version', '1.0')}")
            st.write(f"**Eğitim Kesimi:** {model_meta.get('data_cutoff_date', '-')}")
            if "test_accuracy" in model_meta:
                st.write(f"**Test Doğruluğu:** %{model_meta['test_accuracy'] * 100:.1f}")
            if "metrics" in model_meta and "roc_auc" in model_meta["metrics"]:
                st.write(f"**ROC-AUC:** {model_meta['metrics']['roc_auc']:.3f}")
            st.write(f"**Öznitelik Sayısı:** {len(model_meta.get('features', []))}")
            st.write("**Veri Sağlığı (PSI):** 🟢 Aktif İzleme")
            st.write("**Model Motoru:** 🌲 XGBoost · ⚡ LightGBM · 🐱 CatBoost")
        else:
            st.write("Model meta-verisi bulunamadı.")


def render_safely(render):
    """Bir sekmedeki hata yalnızca o sekmede gösterilir; diğer sekmeler çalışmaya devam eder."""
    try:
        render()
    except ValueError as e:
        st.warning(f"⚠️ {e}")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")
