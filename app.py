# app.py
import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb

from src import (
    backtest,
    config,
    explain,
    live_data,
    model_metadata,
    scanner,
    session_calendar,
    signals,
)

# Sayfa Ayarları
st.set_page_config(page_title="BIST Hisse Yön Tahmini", layout="wide")

st.title("📈 Borsa İstanbul Yapay Zeka Yön Tahmini")
st.markdown("""
Bu proje **XGBoost** algoritması kullanarak BIST 30 hisselerinin
bir sonraki günkü kapanış yönünü (Artış/Düşüş) tahmin eder.
""")

# Yan Menü
st.sidebar.header("Hisse Seçimi")
selected_ticker = st.sidebar.selectbox(
    "Hisse Senedi Seçiniz",
    [t.replace(".IS", "") for t in config.TICKERS],
    format_func=lambda x: (
        f"{x} (TRALT)" if x == "KOZAL" else (f"{x} (TRMET)" if x == "KOZAA" else x)
    ),
)
selected_ticker_full = selected_ticker + ".IS"
# Veriler 15 dk önbellekte tutulur; buton yalnızca seçili hissenin kaydını
# temizler, böylece sonraki çalıştırma Yahoo'dan taze veri çeker (#11)
st.sidebar.button(
    "🔄 Verileri Yenile",
    key="btn_refresh_data",
    help="Seçili hissenin verisini önbelleği atlayarak yeniden indirir.",
    on_click=lambda: get_prediction_data.clear(selected_ticker_full),
)

st.sidebar.header("Sinyal Ayarları")
min_confidence = st.sidebar.slider(
    "Minimum Güven Eşiği",
    min_value=0.51,
    max_value=0.70,
    value=config.PROB_THRESHOLD_HIGH,
    step=0.01,
    help="Olasılık bu eşiğin üzerinde (ya da 1 - eşik altında) değilse sinyal nötr gösterilir.",
)
prob_low, prob_high = signals.symmetric_band(min_confidence)

# Model Bilgisi (#13)
model_meta = model_metadata.load_metadata(config.MODEL_META_PATH)
with st.sidebar.expander("ℹ️ Model Bilgisi", expanded=False):
    if model_meta:
        st.write(f"**Sürüm:** v{model_meta.get('version', '1.0')}")
        st.write(f"**Eğitim Kesimi:** {model_meta.get('data_cutoff_date', '-')}")
        if "test_accuracy" in model_meta:
            st.write(f"**Test Doğruluğu:** %{model_meta['test_accuracy'] * 100:.1f}")
        if "metrics" in model_meta and "roc_auc" in model_meta["metrics"]:
            st.write(f"**ROC-AUC:** {model_meta['metrics']['roc_auc']:.3f}")
        st.write(f"**Öznitelik Sayısı:** {len(model_meta.get('features', []))}")
    else:
        st.write("Model meta-verisi bulunamadı.")


# Model Yükleme
@st.cache_resource
def load_model():
    model = xgb.XGBClassifier()
    model.load_model(config.MODEL_PATH)
    return model


# Canlı Veri Çekme ve İşleme Fonksiyonu
@st.cache_data(ttl=900, show_spinner=False)
def get_prediction_data(ticker):
    # df_processed: tahmin (son satır) ve backtest için; df: grafik çizimi için
    return live_data.fetch_live_frame(ticker)


# Tüm BIST 30 Hisseleri için Önbellekli Tarama Fonksiyonu
@st.cache_data(ttl=900, show_spinner=False)
def get_cached_market_scan(tickers):
    model = load_model()
    return scanner.scan_market(tickers, model)


def render_single_ticker_tab():
    model = load_model()
    # Kullanıcı butona bastığında veya sayfa yüklendiğinde
    with st.spinner(f"{selected_ticker} verileri analiz ediliyor..."):
        df_processed, full_df = get_prediction_data(selected_ticker_full)
        # Sadece en son günü al (Yarın için tahmin yapacağız)
        input_data = df_processed.iloc[[-1]]

        # Gerekli Featurelar
        X_pred = input_data[config.FEATURES]

        # Tahmin
        prob = model.predict_proba(X_pred)[0][1]  # Artış olasılığı
        signal = signals.classify_signal(prob, prob_low, prob_high)
        # Sinyale göre renkli kutu: yeşil (yükseliş), sarı (nötr), kırmızı (düşüş)
        signal_box, signal_label = {
            signals.UP: (st.success, "**YÜKSELİŞ BEKLENTİSİ** 🚀"),
            signals.NEUTRAL: (st.warning, "**NÖTR / BELİRSİZ PİYASA** ⏸️"),
            signals.DOWN: (st.error, "**DÜŞÜŞ / ZAYIF TREND** 🔻"),
        }[signal]

        # Seans Takvimi Bilgilendirmesi (#16)
        session_info = session_calendar.get_session_info(input_data["Date"].iloc[-1])
        if session_info.weekend_notice:
            st.info(session_info.weekend_notice)
        st.caption(session_info.badge_text)

        # GÖSTERGE PANELİ
        col1, col2, col3 = st.columns(3)

        # Fiyat ve Değişim Hesaplama
        if len(full_df) >= 2:
            current_price = full_df["close"].iloc[-1]
            prev_price = full_df["close"].iloc[-2]

            # Değişim Miktarı (TL) ve Oranı (%)
            change_amount = current_price - prev_price
            change_rate = (change_amount / prev_price) * 100
        else:
            current_price = full_df["close"].iloc[-1]
            change_amount = 0
            change_rate = 0

        with col1:
            # Delta color parametresini otomatikte bırakıyoruz, Streamlit +/- algılayıp renk verir
            st.metric(
                label=f"{selected_ticker} Son Fiyat",
                value=f"{current_price:.2f} TL",
                delta=f"{change_rate:.2f}%",  # Format: "-2.15%" veya "1.50%"
            )

        with col2:
            st.write(f"🤖 **Modelin Seans Tahmini ({session_info.next_session_str}):**")
            signal_box(signal_label)

        with col3:
            st.write("📊 **Güven Skoru:**")
            signal_box(f"%{prob * 100:.1f} Olasılıkla Yükseliş")

        # GRAFİK KISMI (Candlestick)
        st.subheader(f"{selected_ticker} - Son 3 Ay Fiyat Grafiği")
        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=full_df["Date"][-90:],
                    open=full_df["open"][-90:],
                    high=full_df["high"][-90:],
                    low=full_df["low"][-90:],
                    close=full_df["close"][-90:],
                )
            ]
        )
        fig.update_layout(xaxis_rangeslider_visible=False)
        st.plotly_chart(fig, width="stretch")

        # Explainability (PDF Şartı: Neden bu karar?)
        # Modelin gerçek TreeSHAP katkıları: her özniteliğin bu tahmindeki payı
        st.subheader("Model Neden Bu Kararı Verdi?")
        explanation, base_prob = explain.explain_prediction(model, X_pred)
        st.markdown(explain.summarize_drivers(explanation, selected_ticker))

        impacts = explanation["impact"] * 100
        levels = base_prob * 100 + impacts.cumsum()
        fig_explain = go.Figure(
            go.Waterfall(
                measure=["absolute"] + ["relative"] * len(explanation) + ["total"],
                x=["Ortalama"]
                + [explain.FEATURE_LABELS.get(f, f) for f in explanation["feature"]]
                + ["Tahmin"],
                y=[base_prob * 100] + impacts.tolist() + [0],
                text=[f"%{base_prob * 100:.1f}"]
                + [f"{i:+.1f}" for i in impacts]
                + [f"%{prob * 100:.1f}"],
                increasing={"marker": {"color": "#2ca02c"}},
                decreasing={"marker": {"color": "#d62728"}},
            )
        )
        # Başlangıç çubuğu %0'dan başlarsa birkaç puanlık katkılar okunmaz; ekseni yakınlaştır
        low, high = min(levels.min(), base_prob * 100), max(levels.max(), base_prob * 100)
        fig_explain.update_layout(
            yaxis_title="Yükseliş Olasılığı (%)",
            yaxis_range=[low - 2, high + 2],
            showlegend=False,
        )
        st.plotly_chart(fig_explain, width="stretch")
        st.caption(
            "Ortalama: modelin hiçbir göstergeyi bilmeden verdiği olasılık. "
            "Her çubuk, ilgili göstergenin bugünkü değerinin olasılığı kaç puan "
            "artırdığını (yeşil) ya da azalttığını (kırmızı) gösterir."
        )

        st.write("Son günün teknik verileri:")
        st.dataframe(input_data[["rsi", "macd", "sma_10", "sma_50", "volatility"]])

        render_backtest(model, df_processed[["Date", "close"] + config.FEATURES])


BACKTEST_MONTHS = 6


def render_backtest(model, df_processed):
    """Modelin yükseliş sinyallerini izleyen al/nakit stratejisini al-tut ile kıyaslar."""
    st.subheader(f"Geçmiş Strateji Getirisi (Backtest, Son {BACKTEST_MONTHS} Ay)")
    window = backtest.recent_window(df_processed, months=BACKTEST_MONTHS)
    probs = model.predict_proba(window.drop(columns=["Date", "close"]))[:, 1]
    result = backtest.run_backtest(window, probs, threshold=prob_high)

    fig = go.Figure()
    for column, name, color in (
        ("strategy", "Model Stratejisi", "#1f77b4"),
        ("buy_hold", "Al ve Tut", "#7f7f7f"),
    ):
        fig.add_trace(
            go.Scatter(
                x=result.curve.index,
                y=result.curve[column] * 100,
                mode="lines",
                name=name,
                line={"color": color},
            )
        )
    fig.update_layout(yaxis_title="Kümülatif Getiri (%)", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    summary = pd.DataFrame(
        [result.strategy, result.buy_hold], index=["Model Stratejisi", "Al ve Tut"]
    )
    table = pd.DataFrame(
        {
            "Toplam Getiri (%)": summary["total_return"] * 100,
            "Sharpe Oranı": summary["sharpe"],
            "Maks. Değer Kaybı (%)": summary["max_drawdown"] * 100,
        }
    )
    st.dataframe(
        table,
        column_config={
            "Toplam Getiri (%)": st.column_config.NumberColumn(format="%.2f"),
            "Sharpe Oranı": st.column_config.NumberColumn(format="%.2f"),
            "Maks. Değer Kaybı (%)": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.caption(
        f"Strateji, yükseliş olasılığı %{prob_high * 100:.0f} ve üzerindeyken hissede kalır, "
        "aksi halde nakde geçer (getiri %0). "
        f"Hissede kalınan gün oranı: %{result.exposure * 100:.0f} · "
        f"Pozisyon değişikliği: {result.trades}. "
        "Komisyon ve vergiler dahil değildir; "
        "geçmiş performans gelecekteki sonuçları garanti etmez."
    )


# Tarama tablolarında sayıların tutarlı ondalıkla gösterimi (örn. 1.9 yerine 1.90)
SCAN_COLUMN_CONFIG = {
    "Son Fiyat (TL)": st.column_config.NumberColumn(format="%.2f"),
    "Günlük Değişim (%)": st.column_config.NumberColumn(format="%.2f"),
    "Yükseliş Olasılığı (%)": st.column_config.NumberColumn(format="%.1f"),
}


def render_market_scanner_tab():
    st.subheader("📊 BIST 30 Piyasa Fırsat Radarı")
    st.markdown("""
    Bu modül BIST 30 endeksindeki tüm hisseleri yapay zeka modelinden geçirerek
    yarın için en yüksek yükseliş potansiyeline ve düşüş riskine sahip hisseleri sıralar.
    """)

    c_btn1, c_btn2 = st.columns([3, 7])
    with c_btn1:
        if st.button("🚀 BIST 30 Taramasını Başlat", key="btn_start_scan"):
            # Yenile butonu aynı çalıştırmada görünsün diye bayrak buton satırından önce set edilir
            st.session_state["market_scan_done"] = True
    with c_btn2:
        if st.session_state.get("market_scan_done", False):
            if st.button("🔄 Taramayı Yenile", key="btn_refresh_scan"):
                get_cached_market_scan.clear()
                st.rerun()

    if st.session_state.get("market_scan_done", False):
        with st.spinner("BIST 30 hisseleri taranıyor ve analiz ediliyor..."):
            scan_df, failed_tickers = get_cached_market_scan(tuple(config.TICKERS))

        if failed_tickers:
            st.warning(
                f"⚠️ {len(failed_tickers)} hisse için veri alınamadı ve taramaya dahil edilmedi: "
                + ", ".join(t.replace(".IS", "") for t in failed_tickers)
            )

        if scan_df is not None and not scan_df.empty:
            top_bull, top_bear = scanner.top_and_bottom(scan_df, n=5)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🟢 En Yüksek Yükseliş Potansiyeli (Top 5)")
                st.dataframe(
                    top_bull[
                        ["Hisse", "Son Fiyat (TL)", "Günlük Değişim (%)", "Yükseliş Olasılığı (%)"]
                    ],
                    hide_index=True,
                    column_config=SCAN_COLUMN_CONFIG,
                )
            with c2:
                st.markdown("### 🔴 Düşüş Riski En Yüksek (Top 5)")
                st.dataframe(
                    top_bear[
                        ["Hisse", "Son Fiyat (TL)", "Günlük Değişim (%)", "Yükseliş Olasılığı (%)"]
                    ],
                    hide_index=True,
                    column_config=SCAN_COLUMN_CONFIG,
                )

            st.markdown("### 📋 Tüm BIST 30 Liderlik Tablosu")
            st.dataframe(
                scan_df[
                    [
                        "Hisse",
                        "Son Fiyat (TL)",
                        "Günlük Değişim (%)",
                        "Yükseliş Olasılığı (%)",
                        "Tahmin",
                    ]
                ],
                hide_index=True,
                column_config=SCAN_COLUMN_CONFIG,
            )
        else:
            st.warning("Piyasa verileri taranırken veri alınamadı.")


def render_safely(render):
    """Bir sekmedeki hata yalnızca o sekmede gösterilir; diğer sekmeler çalışmaya devam eder."""
    try:
        render()
    except ValueError as e:
        st.warning(f"⚠️ {e}")
    except Exception as e:
        st.error(f"Bir hata oluştu: {e}")


# Ana Akış
if not os.path.exists(config.MODEL_PATH):
    st.error("Model dosyası bulunamadı! Lütfen önce `src/model_train.py` çalıştırın.")
else:
    tab1, tab2 = st.tabs(["🎯 Tek Hisse Analizi", "📊 BIST 30 Fırsat Radarı"])

    with tab1:
        render_safely(render_single_ticker_tab)

    with tab2:
        render_safely(render_market_scanner_tab)
