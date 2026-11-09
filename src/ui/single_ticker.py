"""Tek hisse senedi detay analizi, SHAP açıklamaları, grafikler ve backtest sekmesi."""

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import (
    backtest,
    config,
    drift,
    explain,
    fundamental_rag,
    kap_llm,
    kap_news,
    risk_sizing,
    session_calendar,
    signals,
)
from src.ui.components import get_prediction_data, load_model

BACKTEST_MONTHS = 6


def render_backtest(model, df_processed, threshold=config.PROB_THRESHOLD_HIGH, cost=0.0):
    """Modelin yükseliş sinyallerini izleyen al/nakit stratejisini al-tut ile kıyaslar."""
    st.subheader(f"Geçmiş Strateji Getirisi (Backtest, Son {BACKTEST_MONTHS} Ay)", anchor=False)
    use_kelly = st.checkbox(
        "🎯 Dinamik Kelly Boyutlandırmalı Backtest Göster",
        value=False,
        help="Sermayeyi sabit %100 yerine güven skoru ve oynaklığa göre dinamik tahsis eder (#71).",
    )
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

    result_kelly = None
    if use_kelly:
        result_kelly = backtest.run_backtest(
            window, probs, threshold=threshold, cost=cost, sizing="kelly"
        )
        fig.add_trace(
            go.Scatter(
                x=result_kelly.curve.index,
                y=result_kelly.curve["strategy"] * 100,
                mode="lines",
                name="Model (Kelly Dinamik)",
                line={"color": "#2ca02c", "dash": "dot"},
            )
        )

    fig.update_layout(yaxis_title="Kümülatif Getiri (%)", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

    if use_kelly and result_kelly is not None:
        summary_rows = [result.strategy, result_kelly.strategy, result.buy_hold]
        indices = ["Model Stratejisi (Sabit)", "Model (Kelly Dinamik)", "Al ve Tut"]
    else:
        summary_rows = [result.strategy, result.buy_hold]
        indices = ["Model Stratejisi", "Al ve Tut"]

    summary = pd.DataFrame(summary_rows, index=indices)
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

    with st.expander("💡 Bu Kıyaslama Grafiği Nasıl Okunmalı?", expanded=False):
        st.markdown(
            """
            * **🔘 Al ve Tut (Gri Çizgi - Benchmark):** Hissenin **kendi saf piyasa getirisi ve
              doğal fiyat grafiğidir**. 6 ay önce hisseyi alıp aradaki hiçbir düşüş veya yükselişe
              dokunmadan bugüne kadar öylece tutsaydınız portföyün nasıl değişeceğini gösterir.
            * **🤖 Model Stratejisi (Mavi Çizgi):** Yapay zekanın yönettiği
              **alım-satım portföyüdür**. Model *Yükseliş* öngördüğünde hissede kalarak kârı
              hedefler; *Düşüş* veya *Belirsiz* öngördüğünde ise hisseyi satıp **TL Nakde** geçer.
              Amaç, sert piyasa düşüşlerinden nakitte kalarak korunmak ve yalnızca yükseliş
              dalgalarından getiri sağlamaktır.
            * **🎯 Model (Kelly Dinamik - Yeşil Kesikli Çizgi):** Pozisyon boyutunu sabit %100
              yapmak yerine güven skoru ve oynaklığa (volatilite) göre dinamik ayarlayan
              risk kontrollü sermaye dağılımıdır.
            * **📌 Özet Kuralı:** Mavi çizgi gri çizginin üzerinde kaldığı sürece; model düşüş
              günlerinde nakde geçerek hisseye göre **daha az kaybetmiş veya daha çok kazanmış**
              demektir.
            """
        )


def render_risk_sizing_section(selected_ticker: str, prob: float, volatility: float):
    """Kelly Kriteri ve volatiliteye duyarlı pozisyon boyutlandırma kartı (#71)."""
    with st.expander("🎯 Sermaye Tahsisi & Pozisyon Boyutlandırma (Kelly)", expanded=False):
        rec = risk_sizing.recommend_position_size(
            selected_ticker,
            prob=prob,
            volatility=volatility,
            min_threshold=config.PROB_THRESHOLD_HIGH,
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**Önerilen Hisse Payı:** `%{rec.position_size * 100:.1f}`")
        c2.markdown(f"**Önerilen Nakit Payı:** `%{rec.cash_size * 100:.1f}`")
        c3.markdown(f"**Risk Seviyesi:** {rec.risk_level}")

        st.caption(
            f"ℹ️ **Hesaplama:** Ham Kelly: `%{rec.kelly_raw * 100:.1f}` · "
            f"Volatilite Çarpanı: `{rec.vol_multiplier:.2f}` "
            f"(Hedef: %25.0, Hisse: %{volatility * 100:.1f})"
        )
        st.info(rec.summary_note)


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


def render_kap_llm_section(selected_ticker: str):
    """KAP bildirimlerini ve LLM destekli temel analiz özetini gösterir (#70)."""
    with st.expander("🤖 Yapay Zeka Temel Analiz & KAP Yorumu", expanded=False):
        disclosures = kap_news.fetch_latest_disclosures(selected_ticker, limit=5)
        analysis = kap_llm.analyze_disclosures_with_llm(selected_ticker, disclosures)

        provider_badge = (
            "🤖 Gemini LLM"
            if analysis.provider == "gemini"
            else "⚙️ Kural Tabanlı (Yedek Mod)"
            if analysis.provider == "rule_based"
            else "⚪ Bildirim Yok"
        )
        c1, c2, c3 = st.columns(3)
        c1.markdown(f"**KAP Etki Rozeti:** {analysis.badge}")
        c2.markdown(f"**Etki Skoru:** `{analysis.impact_score:+.2f}`")
        c3.markdown(f"**Analiz Kaynağı:** `{provider_badge}`")

        if analysis.summary:
            st.markdown(f"**Özet Değerlendirme:** {analysis.summary}")

        st.markdown("**Finansal Öne Çıkanlar (Temel Göstergeler):**")
        for bullet in analysis.bullets:
            st.markdown(f"- {bullet}")

        if analysis.error_message:
            st.caption(f"ℹ️ Not: {analysis.error_message}")

        if disclosures:
            st.markdown("---")
            st.markdown("**Son KAP Bildirimleri & Şirket Haberleri:**")
            disc_table = pd.DataFrame(
                [
                    {
                        "Tarih": d.publish_date,
                        "Kategori": d.badge,
                        "Başlık": d.title,
                    }
                    for d in disclosures
                ]
            )
            st.dataframe(disc_table)


def render_fundamental_rag_section(selected_ticker: str):
    """Financial Analyst ChatBot Project temel analiz ve bilanço özetini gösterir."""
    with st.expander("💬 Kurumsal Temel Analiz & Bilanço Chatbot'u (RAG)", expanded=False):
        report = fundamental_rag.fetch_fundamental_report(selected_ticker)
        chatbot_link = fundamental_rag.build_chatbot_deeplink(selected_ticker)

        box_style = (
            "background-color: #1E293B; border: 1px solid #334155; "
            "border-radius: 8px; padding: 12px 16px; margin-bottom: 12px; "
            "display: flex; justify-content: space-between; align-items: center; "
            "flex-wrap: wrap; gap: 8px;"
        )
        btn_style = (
            "background-color: #2563EB; color: #FFFFFF; font-weight: 600; "
            "font-size: 0.82rem; padding: 6px 14px; border-radius: 6px; "
            "text-decoration: none; display: inline-block;"
        )
        st.markdown(
            f"""
            <div style="{box_style}">
                <span style="color: #F8FAFC; font-size: 0.90rem;">
                    🏢 <strong>{selected_ticker}</strong> için KAP raporları, bilanço ve dipnotları
                    analiz eden yapay zeka asistanına soru sorun.
                </span>
                <a href="{chatbot_link}" target="_blank" style="{btn_style}">
                    Chatbot'a Git ↗
                </a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if report is not None:
            c1, c2, c3 = st.columns(3)
            c1.markdown(f"**Temel Görünüm:** {report.badge}")
            c2.markdown(f"**Rapor Tarihi:** `{report.report_date or 'Güncel'}`")
            source_badge = (
                "⚡ Yerel Önbellek"
                if report.source == "local_cache"
                else ("🌐 Chatbot API" if report.source == "api" else "🐙 GitHub Veri Deposu")
            )
            c3.markdown(f"**Veri Kaynağı:** `{source_badge}`")

            if report.fundamental_summary:
                st.markdown(f"**Bilanço & Faaliyet Özeti:**\n\n{report.fundamental_summary}")

            km = report.key_metrics
            r_col1, r_col2, r_col3, r_col4 = st.columns(4)
            pe_str = f"{km.pe_ratio:.2f}" if km.pe_ratio is not None else "—"
            pb_str = f"{km.pb_ratio:.2f}" if km.pb_ratio is not None else "—"
            r_col1.markdown(f"**F/K Oranı:** `{pe_str}`")
            r_col2.markdown(f"**PD/DD:** `{pb_str}`")
            r_col3.markdown(f"**Net Kâr Trendi:** `{km.net_profit_margin_trend}`")
            r_col4.markdown(f"**Borçluluk:** `{km.debt_to_equity}`")

            if report.catalysts:
                st.markdown("**🚀 Yukarı Yönlü Katalizörler & Büyüme Faktörleri:**")
                for cat in report.catalysts:
                    st.markdown(f"- {cat}")

            if report.risk_factors:
                st.markdown("**⚠️ Temel ve Finansal Risk Faktörleri:**")
                for risk in report.risk_factors:
                    st.markdown(f"- {risk}")
        else:
            st.markdown(
                f"ℹ️ *{selected_ticker} için önceden üretilmiş statik özet rapor bulunamadı. "
                f"Yukarıdaki butona tıklayarak **Financial Analyst ChatBot** üzerinden "
                f"canlı RAG analizini başlatabilirsiniz.*"
            )


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

        # Sermaye Tahsisi & Pozisyon Boyutlandırma (Kelly) (#71)
        vol = (
            float(input_data["volatility"].iloc[-1]) if "volatility" in input_data.columns else 0.25
        )
        render_risk_sizing_section(selected_ticker, prob, vol)

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
        render_kap_llm_section(selected_ticker)
        render_fundamental_rag_section(selected_ticker)
