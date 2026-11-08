"""Tek hisse senedi detay analizi, SHAP açıklamaları, grafikler ve backtest sekmesi."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import backtest, config, drift, explain, session_calendar, signals
from src.ui.components import get_prediction_data, load_model

BACKTEST_MONTHS = 6


def render_backtest(model, df_processed, threshold=config.PROB_THRESHOLD_HIGH, cost=0.0):
    """Modelin yükseliş sinyallerini izleyen al/nakit stratejisini al-tut ile kıyaslar."""
    st.subheader(f"Geçmiş Strateji Getirisi (Backtest, Son {BACKTEST_MONTHS} Ay)", anchor=False)
    window = backtest.recent_window(df_processed, months=BACKTEST_MONTHS)
    probs = model.predict_proba(window.drop(columns=["Date", "close"]))[:, 1]
    result = backtest.run_backtest(window, probs, threshold=threshold, cost=cost)

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
            "Kazanma Oranı (%)": summary["win_rate"] * 100,
            "Kâr Faktörü": summary["profit_factor"],
        }
    )
    st.dataframe(
        table,
        column_config={
            "Toplam Getiri (%)": st.column_config.NumberColumn(format="%.2f"),
            "Sharpe Oranı": st.column_config.NumberColumn(format="%.2f"),
            "Maks. Değer Kaybı (%)": st.column_config.NumberColumn(format="%.2f"),
            "Kazanma Oranı (%)": st.column_config.NumberColumn(format="%.1f"),
            "Kâr Faktörü": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    cost_text = (
        f"İşlem başına %{cost * 100:.2f} komisyon ve kayma maliyeti yansıtılmıştır · "
        if cost > 0
        else "Komisyon ve vergiler dahil değildir · "
    )
    st.caption(
        f"Strateji, yükseliş olasılığı %{threshold * 100:.0f} ve üzerindeyken hissede kalır, "
        "aksi halde nakde geçer (getiri %0). "
        f"Hissede kalınan gün oranı: %{result.exposure * 100:.0f} · "
        f"Pozisyon değişikliği: {result.trades} · "
        f"{cost_text}"
        "Geçmiş performans gelecekteki sonuçları garanti etmez."
    )


def render_drift_section(df_features):
    """Öznitelik sapmasını (Data Drift / PSI) hesaplayıp özetler (#45, #69)."""
    with st.expander("📊 Veri & Öznitelik Sapması (Data Drift / PSI)", expanded=False):
        drift_df, summary = drift.compute_series_drift_by_split(df_features, recent_rows=60)
        cyclical_text = (
            f" · Takvim Döngüsü: `{summary['max_cyclical_psi']:.2f}`"
            if summary.get("max_cyclical_psi", 0.0) > 0.25
            else ""
        )
        st.markdown(
            f"**Genel Sağlık Durumu:** {summary['badge']} | "
            f"**Ortalama Gösterge PSI:** `{summary['mean_psi']:.4f}` | "
            f"**Maksimum Gösterge PSI:** `{summary['max_psi']:.4f}`"
            f"{cyclical_text}"
        )
        st.caption(
            "Population Stability Index (PSI), geçmiş referans dönem ile son 60 günün "
            "teknik gösterge dağılımlarını kıyaslar. Takvim öznitelikleri doğal döngü olarak "
            "ayrı etiketlenir ve gösterge sapma puanlarını etkilemez. "
            "PSI < 0.10: Sağlıklı, 0.10-0.25: Hafif Sapma, > 0.25: Kritik Sapma."
        )
        if not drift_df.empty:
            st.dataframe(drift_df[["Öznitelik", "Rozet", "PSI", "Öneri"]])


def render_single_ticker_tab(
    selected_ticker: str,
    selected_ticker_full: str,
    prob_low: float,
    prob_high: float,
    cost: float = 0.0,
):
    """Tek hisse senedi detay analizi sekmesini oluşturur."""
    model = load_model()
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
            st.metric(
                label=f"{selected_ticker} Son Fiyat",
                value=f"{current_price:.2f} TL",
                delta=f"{change_rate:.2f}%",
            )

        with col2:
            st.write(f"🤖 **Modelin Seans Tahmini ({session_info.next_session_str}):**")
            signal_box(signal_label)

        with col3:
            st.write("📊 **Güven Skoru:**")
            signal_box(f"%{prob * 100:.1f} Olasılıkla Yükseliş")

        # Makro ve Piyasa Göstergeleri (#10)
        macro_cols = ["xu100_ret", "rel_strength_bist", "usdtry_change"]
        if all(col in input_data.columns for col in macro_cols):
            with st.expander("🌐 Makro Piyasa Göstergeleri (BIST 100 & USD/TRY)", expanded=False):
                m1, m2, m3 = st.columns(3)
                xu_pct = input_data["xu100_ret"].iloc[-1] * 100
                rel_pct = input_data["rel_strength_bist"].iloc[-1] * 100
                fx_pct = input_data["usdtry_change"].iloc[-1] * 100
                m1.write(f"**🏛️ BIST 100 Getirisi:** %{xu_pct:+.2f}")
                m2.write(f"**💪 BIST 100 Rölatif Güç:** %{rel_pct:+.2f}")
                m3.write(f"**💵 USD/TRY Değişimi:** %{fx_pct:+.2f}")

        # GRAFİK KISMI (Candlestick)
        st.subheader(f"{selected_ticker} - Son 3 Ay Fiyat Grafiği", anchor=False)
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

        # Explainability (TreeSHAP)
        st.subheader("Model Neden Bu Kararı Verdi?", anchor=False)
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

        render_backtest(
            model,
            df_processed[["Date", "close"] + config.FEATURES],
            threshold=prob_high,
            cost=cost,
        )
        render_drift_section(df_processed[config.FEATURES])
