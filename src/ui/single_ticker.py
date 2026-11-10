"""Tek hisse senedi detay analizi, SHAP açıklamaları, grafikler ve backtest sekmesi."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import (
    backtest,
    colab_llm,
    config,
    drift,
    explain,
    features,
    fundamental_rag,
    hybrid_decision,
    kap_llm,
    kap_news,
    live_market,
    multi_horizon,
    patterns,
    risk_sizing,
    session_calendar,
    signals,
    simulator,
    tracker,
)
from src.ui.components import get_prediction_data, load_model

BACKTEST_MONTHS = 6

PLOTLY_CONFIG: dict[str, object] = {
    "scrollZoom": True,
    "displayModeBar": "hover",
    "doubleClick": "reset+autosize",
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


def render_backtest(model, df_processed, threshold=config.PROB_THRESHOLD_HIGH, cost=0.0):
    """Modelin yükseliş sinyallerini izleyen al/nakit stratejisini al-tut ile kıyaslar."""
    st.subheader(f"Geçmiş Strateji Getirisi (Backtest, Son {BACKTEST_MONTHS} Ay)", anchor=False)
    use_kelly = st.checkbox(
        "🎯 Dinamik Kelly Boyutlandırmalı Backtest Göster",
        value=False,
        help="Sermayeyi sabit %100 yerine güven skoru ve oynaklığa göre dinamik tahsis eder (#71).",
    )
    window = backtest.recent_window(df_processed, months=BACKTEST_MONTHS)
    probs = model.predict_proba(features.align_model_features(model, window))[:, 1]
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

    fig.update_layout(
        yaxis_title="Kümülatif Getiri (%)",
        hovermode="x unified",
        dragmode="pan",
    )
    st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

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
            * **📈 Model Stratejisi (Mavi Çizgi):** Yapay zekanın yönettiği
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
    st.subheader("🔬 Veri & Öznitelik Sapması (Data Drift / PSI)", anchor=False)
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


@st.cache_data(ttl=900, show_spinner=False)
def get_cached_fundamental_report(ticker: str) -> fundamental_rag.FundamentalReport | None:
    """Tekil hisse için temel analiz raporunu 15 dakika boyunca önbellekte tutar."""
    return fundamental_rag.fetch_fundamental_report(ticker)


@st.cache_data(ttl=30, show_spinner=False)
def get_cached_live_market_quotes() -> dict[str, live_market.LiveQuote]:
    """BIST hisselerinin anlık canlı kotasyonlarını 30 saniye boyunca önbellekte tutar."""
    return live_market.fetch_live_quotes()


def render_kap_llm_section(
    selected_ticker: str,
    report: fundamental_rag.FundamentalReport | None = None,
):
    """KAP bildirimlerini ve LLM destekli temel analiz özetini gösterir (#70)."""
    st.subheader("📑 Yapay Zeka Temel Analiz & KAP Yorumu", anchor=False)
    disclosures = kap_news.fetch_latest_disclosures(selected_ticker, limit=5)
    is_rag_source = False
    has_signal = any(d.category != kap_news.EventCategory.NEUTRAL for d in disclosures)
    if (not disclosures or not has_signal) and report is not None and report.recent_disclosures:
        disclosures = fundamental_rag.convert_disclosures_to_kap(
            selected_ticker, report.recent_disclosures
        )
        is_rag_source = True

    analysis = kap_llm.analyze_disclosures_with_llm(selected_ticker, disclosures)

    if is_rag_source:
        if analysis.provider == "gemini":
            provider_badge = "✨ Gemini LLM (RAG KAP)"
        else:
            provider_badge = "🏢 RAG Chatbot KAP Verisi"
    else:
        provider_badge = (
            "✨ Gemini LLM"
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

    if not disclosures and report is not None and report.fundamental_summary:
        pe_val = f"{report.key_metrics.pe_ratio:.2f}" if report.key_metrics.pe_ratio else "—"
        pb_val = f"{report.key_metrics.pb_ratio:.2f}" if report.key_metrics.pb_ratio else "—"
        st.info(
            f"💡 **Kurumsal Bilanço Köprüsü:** Son günlerde olağanüstü anlık bir KAP bildirimi "
            f"(ihale, SPK tedbiri, geri alım vb.) bulunmamaktadır. Şirketin güncel dönem "
            f"**Bilanço Görünümü: {report.badge}** (F/K: `{pe_val}`, PD/DD: `{pb_val}`).\n\n"
            f"Dönemsel kârlılık, büyüme katalizörleri ve risk faktörleri için sol menüdeki "
            f"**'🏢 Kurumsal Bilanço & Chatbot'** sekmesini inceleyebilirsiniz."
        )

    if analysis.error_message:
        st.caption(f"ℹ️ Not: {analysis.error_message}")

    if is_rag_source and report is not None and report.recent_disclosures:
        st.markdown("---")
        st.markdown("**Son KAP & Finansal Bildirimleri (RAG Chatbot Veri Deposu):**")
        formatted_rows = []
        for item in report.recent_disclosures:
            etki_val = str(item.get("Etki", item.get("impact", "Nötr")))
            if any(pos in etki_val.lower() for pos in ["pozitif", "olumlu", "positive"]):
                etki_badge = "🟢 Pozitif"
            elif any(neg in etki_val.lower() for neg in ["negatif", "olumsuz", "negative"]):
                etki_badge = "🔴 Negatif"
            else:
                etki_badge = "⚪ Nötr"
            formatted_rows.append(
                {
                    "Tarih": item.get("Tarih", item.get("date", "—")),
                    "Kategori": item.get("Kategori", item.get("category", "KAP")),
                    "Başlık": item.get("Başlık", item.get("title", "—")),
                    "Etki": etki_badge,
                }
            )
        st.table(pd.DataFrame(formatted_rows))
    elif disclosures:
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


def render_fundamental_rag_section(
    selected_ticker: str,
    report: fundamental_rag.FundamentalReport | None = None,
):
    """Financial Analyst ChatBot Project temel analiz ve bilanço özetini gösterir."""
    st.subheader("🏢 Kurumsal Temel Analiz & Bilanço Chatbot'u (RAG)", anchor=False)
    if report is None:
        report = get_cached_fundamental_report(selected_ticker)
    chatbot_link = fundamental_rag.build_chatbot_deeplink(selected_ticker)

    box_style = (
        "background-color: #0F131D; border: 1px solid #1A2233; "
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

        if report.recent_disclosures:
            st.markdown("---")
            st.markdown("**📋 Son KAP & Finansal Bildirimleri (Chatbot Hafızası):**")
            formatted_disclosures = []
            for item in report.recent_disclosures:
                etki_val = str(item.get("Etki", item.get("impact", "Nötr")))
                if any(pos in etki_val.lower() for pos in ["pozitif", "olumlu", "positive"]):
                    etki_badge = "🟢 Pozitif"
                elif any(neg in etki_val.lower() for neg in ["negatif", "olumsuz", "negative"]):
                    etki_badge = "🔴 Negatif"
                else:
                    etki_badge = "⚪ Nötr"
                formatted_disclosures.append(
                    {
                        "Tarih": item.get("Tarih", item.get("date", "—")),
                        "Kategori": item.get("Kategori", item.get("category", "KAP")),
                        "Başlık": item.get("Başlık", item.get("title", "—")),
                        "Etki": etki_badge,
                    }
                )
            st.table(pd.DataFrame(formatted_disclosures))
    else:
        st.markdown(
            f"ℹ️ *{selected_ticker} için önceden üretilmiş statik özet rapor bulunamadı. "
            f"Yukarıdaki butona tıklayarak **Financial Analyst ChatBot** üzerinden "
            f"canlı RAG analizini başlatabilirsiniz.*"
        )

    # 🚀 Google Colab Pro Fine-Tuned Canlı BIST LLM Entegrasyonu
    st.markdown("---")
    st.markdown("#### 🚀 Canlı Fine-Tuned BIST LLM (Google Colab Pro GPU — A100)")
    st.caption(
        "Kendi eğittiğimiz `bist-financial-llm` modeli Colab üzerinde çalışmakta ve "
        "doğrudan BIST 100 bilanço & KAP eğitim verileriyle yanıt vermektedir."
    )
    colab_active = colab_llm.check_colab_health()
    colab_badge = (
        "🟢 Çevrimiçi (Colab ngrok Aktif: bist-financial-llm)"
        if colab_active
        else "⚪ Çevrimdışı (ngrok bekleniyor)"
    )
    st.markdown(f"**Model Sunucusu Durumu:** `{colab_badge}`")

    c_prompt, c_ask = st.columns([8, 2])
    default_prompt = (
        f"{selected_ticker} için 2024 yılı finansal karne, DuPont ROE ve riskleri özetle."
    )
    user_q = c_prompt.text_input(
        "Soru:",
        value=default_prompt,
        key=f"colab_q_{selected_ticker}",
        label_visibility="collapsed",
    )
    if c_ask.button("✨ Sor", key=f"btn_ask_colab_{selected_ticker}"):
        with st.spinner("Fine-Tuned BIST modeli yanıt üretiyor..."):
            answer = colab_llm.query_colab_llm(user_q)
            st.markdown(f"**✨ Model Değerlendirmesi:**\n\n{answer}")


def render_hybrid_decision_card(decision: hybrid_decision.HybridDecision) -> None:
    """Teknik ve temel analizi birleştiren hibrit karar rozetini ve özetini render eder."""
    verdict_colors = {
        "STRONG_BUY": {
            "border": "#059669",
            "badge_bg": "rgba(5, 150, 105, 0.15)",
            "text": "#34D399",
        },
        "VALUE_DIP": {
            "border": "#1D4ED8",
            "badge_bg": "rgba(29, 78, 216, 0.15)",
            "text": "#60A5FA",
        },
        "SPECULATIVE_MOMENTUM": {
            "border": "#D97706",
            "badge_bg": "rgba(217, 119, 6, 0.15)",
            "text": "#FBBF24",
        },
        "STRONG_SELL": {
            "border": "#B91C1C",
            "badge_bg": "rgba(185, 28, 28, 0.20)",
            "text": "#F87171",
        },
        "TECHNICAL_UP": {
            "border": "#059669",
            "badge_bg": "rgba(5, 150, 105, 0.15)",
            "text": "#34D399",
        },
        "TECHNICAL_DOWN": {
            "border": "#DC2626",
            "badge_bg": "rgba(220, 38, 38, 0.15)",
            "text": "#F87171",
        },
    }
    theme = verdict_colors.get(
        decision.verdict,
        {"border": "#1A2233", "badge_bg": "rgba(26, 34, 51, 0.25)", "text": "#94A3B8"},
    )

    fund_text = (
        f"{decision.fundamental_score:+.2f}"
        if decision.fundamental_score is not None
        else "Rapor Yok"
    )

    card_style = (
        f"background-color: #0F131D; border-left: 5px solid {theme['border']}; "
        "border-radius: 8px; padding: 14px 18px; margin: 14px 0 16px 0; "
        "border-top: 1px solid #1A2233; border-right: 1px solid #1A2233; "
        "border-bottom: 1px solid #1A2233;"
    )
    badge_style = (
        f"background-color: {theme['badge_bg']}; color: {theme['text']}; "
        f"font-size: 0.78rem; font-weight: 700; padding: 3px 8px; border-radius: 4px; "
        f"border: 1px solid {theme['border']};"
    )
    meta_style = (
        "font-size: 0.78rem; color: #64748B; border-top: 1px solid #1A2233; "
        "padding-top: 8px; display: flex; gap: 16px; flex-wrap: wrap;"
    )
    header_style = (
        "display: flex; justify-content: space-between; align-items: center; "
        "flex-wrap: wrap; gap: 8px; margin-bottom: 8px;"
    )

    card_html = f"""
    <div style="{card_style}">
        <div style="{header_style}">
            <div style="display: flex; align-items: center; gap: 10px;">
                <span style="font-size: 1.02rem; font-weight: 700; color: #FFFFFF;">
                    {decision.badge}
                </span>
                <span style="{badge_style}">
                    Kompozit Skor: {decision.composite_score:.1f} / 100
                </span>
            </div>
            <div style="font-size: 0.82rem; color: #94A3B8;">
                Uyum Düzeyi: <strong style="color: #F8FAFC;">{decision.alignment}</strong>
            </div>
        </div>
        <p style="color: #CBD5E1; font-size: 0.88rem; line-height: 1.45; margin: 0 0 10px 0;">
            {decision.summary}
        </p>
        <div style="{meta_style}">
            <span>📈 <strong>Teknik Olasılık:</strong> %{decision.technical_prob * 100:.1f}</span>
            <span>📑 <strong>KAP Temel Duygu:</strong> {fund_text}</span>
            <span>⚖️ <strong>Ağırlık:</strong> %55 Teknik + %45 Temel</span>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_forward_tracking_section(selected_ticker: str, full_df: pd.DataFrame):
    """Kayıtlı tahminlerin gerçekleşen piyasa fiyatlarıyla başarı karnesini gösterir (#53)."""
    st.subheader(
        "🎯 Gerçekleşen Model Başarısı & İleriye Dönük Takip (Forward Tracking)",
        anchor=False,
    )
    all_preds = tracker.load_predictions()
    ticker_preds = [p for p in all_preds if p.get("ticker") == selected_ticker]

    if not ticker_preds:
        st.info(
            f"ℹ️ {selected_ticker} için ilk tahmin kaydı bugünkü seans ile oluşturuldu. "
            "Sonraki seans kapanışlarında gerçekleşen getiri ve isabet oranları bu "
            "tabloda otomatik güncellenecektir."
        )
        return

    evaluated_df = tracker.evaluate_realization(ticker_preds, full_df, horizons=(1, 3, 5))
    if evaluated_df.empty:
        st.info(f"ℹ️ {selected_ticker} için geçmiş seans gerçekleşmeleri hesaplanıyor...")
        return

    summary = tracker.summarize_performance(evaluated_df, horizons=(1, 3, 5))
    horizons_info = summary.get("horizons", {})

    # KPI Rozetleri (len(at.metric) == 1 kuralını korumak için HTML ile)
    kpi_cols = st.columns(3)
    labels = [("1G", "1 Seans"), ("3G", "3 Seans"), ("1H (5G)", "1 Hafta (5 Seans)")]

    badge_box_style = (
        "background-color: #0F131D; border: 1px solid #1A2233; border-radius: 6px; "
        "padding: 10px 14px; text-align: center; margin-bottom: 12px;"
    )

    for col, (h_key, h_title) in zip(kpi_cols, labels):
        h_data = horizons_info.get(h_key, {})
        completed = h_data.get("completed_count", 0)
        hit_rate = h_data.get("hit_rate_pct", 0.0)
        avg_ret = h_data.get("avg_return_pct", 0.0)

        ret_color = "#059669" if avg_ret >= 0 else "#DC2626"
        title_div = (
            f'<div style="font-size: 0.75rem; color: #94A3B8; font-weight: 600;">'
            f"{h_title} İsabet</div>"
        )
        if completed > 0:
            rate_div = (
                f'<div style="font-size: 1.25rem; font-weight: 800; color: #F8FAFC; '
                f'margin: 2px 0;">%{hit_rate:.1f}</div>'
            )
            ret_div = (
                f'<div style="font-size: 0.75rem; color: {ret_color};">'
                f"Ort. Getiri: %{avg_ret:+.2f} ({completed} seans)</div>"
            )
            html = f'<div style="{badge_box_style}">{title_div}{rate_div}{ret_div}</div>'
        else:
            pending_div = (
                '<div style="font-size: 1.05rem; font-weight: 700; color: #F59E0B; '
                'margin: 4px 0;">⏳ Bekleniyor</div>'
            )
            wait_div = '<div style="font-size: 0.75rem; color: #64748B;">Seans tamamlanmadı</div>'
            html = f'<div style="{badge_box_style}">{title_div}{pending_div}{wait_div}</div>'
        col.markdown(html, unsafe_allow_html=True)

    st.caption(
        "Modelin geçmişte verdiği yön tahminlerinin 1 gün, 3 gün ve 1 hafta (5 işlem günü) "
        "sonraki gerçekleşen fiyat karşılıkları ve başarı karnesi:"
    )

    display_df = evaluated_df.iloc[::-1].reset_index(drop=True)
    st.dataframe(
        display_df,
        column_config={
            "Giriş Fiyatı (TL)": st.column_config.NumberColumn(format="%.2f"),
            "1G Fiyat (TL)": st.column_config.NumberColumn(format="%.2f"),
            "1G Getiri (%)": st.column_config.NumberColumn(format="%+.2f"),
            "3G Fiyat (TL)": st.column_config.NumberColumn(format="%.2f"),
            "3G Getiri (%)": st.column_config.NumberColumn(format="%+.2f"),
            "1H (5G) Fiyat (TL)": st.column_config.NumberColumn(format="%.2f"),
            "1H (5G) Getiri (%)": st.column_config.NumberColumn(format="%+.2f"),
            "1G 100k TL Bakiye": st.column_config.NumberColumn(format="%.2f TL"),
            "3G 100k TL Bakiye": st.column_config.NumberColumn(format="%.2f TL"),
            "1H (5G) 100k TL Bakiye": st.column_config.NumberColumn(format="%.2f TL"),
        },
    )


def render_multi_horizon_matrix(
    selected_ticker: str,
    input_data: pd.DataFrame,
    full_df: pd.DataFrame,
    current_price: float,
    prob_low: float,
    prob_high: float,
) -> None:
    """1 Gün, 1 Hafta (5G) ve 1 Ay (20G) vadeli tahmin matrisi ve trend uyumunu render eder."""
    models = multi_horizon.load_multi_horizon_models()
    horizon_res = multi_horizon.predict_horizons(
        models,
        input_data,
        prob_low=prob_low,
        prob_high=prob_high,
    )

    # Yüzde getiri oynaklığı üzerinden hesapla
    if full_df is not None and len(full_df) >= 10:
        vol = float(full_df["close"].pct_change().tail(20).std())
        if np.isnan(vol) or vol <= 0:
            vol = 0.02
    else:
        vol = 0.02
    corridors = multi_horizon.calculate_expected_price_corridor(current_price, vol)

    st.markdown("#### 🎯 Çoklu Vade Trend Matrisi (1 Gün • 1 Hafta • 1 Ay)")

    # 3 Vade Kartı (Derin Obsidyen Kurumsal Palet: #0F131D zemin, #1A2233 sınır)
    m_col1, m_col2, m_col3 = st.columns(3)
    cards = [
        ("1d", "📅 Yarınki Seans (1 Gün)", m_col1),
        ("5d", "📅 Bu Hafta (5 Gün)", m_col2),
        ("20d", "📅 Bu Ay (20 Gün)", m_col3),
    ]

    for key, title, col in cards:
        fc = horizon_res.forecasts.get(key)
        corr = corridors.get(
            key, {"min": current_price * 0.98, "max": current_price * 1.02, "pct": 2.0}
        )
        if fc is None:
            continue

        if fc.signal == 1:
            bg_badge = "rgba(5, 150, 105, 0.18)"
            border_color = "#059669"
            text_color = "#34D399"
            icon = "🚀"
        elif fc.signal == -1:
            bg_badge = "rgba(220, 38, 38, 0.18)"
            border_color = "#DC2626"
            text_color = "#F87171"
            icon = "🔻"
        else:
            bg_badge = "rgba(217, 119, 6, 0.18)"
            border_color = "#D97706"
            text_color = "#FBBF24"
            icon = "⏸️"

        card_html = (
            f'<div style="background-color: #0F131D; border: 1px solid #1E293B; '
            f"border-radius: 10px; padding: 14px 16px; margin-bottom: 12px; "
            f'border-top: 3px solid {border_color}; box-shadow: 0 4px 12px rgba(0,0,0,0.25);">'
            f'<div style="font-size: 0.82rem; color: #94A3B8; font-weight: 600; '
            f'margin-bottom: 8px;">{title}</div>'
            f'<div style="display: flex; justify-content: space-between; '
            f'align-items: flex-end; margin-bottom: 10px;">'
            f'<div><div style="font-size: 0.70rem; color: #94A3B8; text-transform: uppercase; '
            f'font-weight: 600; letter-spacing: 0.5px;">Yükseliş İhtimali</div>'
            f'<div style="font-size: 1.35rem; font-weight: 800; color: #F8FAFC; line-height: 1.1;">'
            f"%{fc.probability * 100:.1f}</div></div>"
            f'<span style="background-color: {bg_badge}; color: {text_color}; '
            f"font-size: 0.78rem; font-weight: 700; padding: 4px 10px; "
            f'border-radius: 6px; border: 1px solid {border_color};">'
            f"{icon} {fc.label}</span></div>"
            f'<div style="border-top: 1px solid #1E293B; padding-top: 8px; margin-top: 4px;">'
            f'<div style="display: flex; justify-content: space-between; '
            f'align-items: center; margin-bottom: 4px;">'
            f'<span style="font-size: 0.82rem; color: #E2E8F0; font-weight: 600;">'
            f"🛡️ %95 Fiyat Koridoru</span>"
            f'<span style="font-size: 0.76rem; color: #38BDF8; font-weight: 700; '
            f"background: rgba(56, 189, 248, 0.12); padding: 1px 7px; border-radius: 4px; "
            f'border: 1px solid rgba(56, 189, 248, 0.25);">±%{corr["pct"]:.1f}</span></div>'
            f'<div style="font-size: 0.95rem; font-weight: 700; color: #F8FAFC; '
            f'letter-spacing: 0.3px;">{corr["min"]:.2f} TL '
            f'<span style="color: #64748B; font-weight: 400; font-size: 0.85rem;">—</span> '
            f"{corr['max']:.2f} TL</div></div></div>"
        )
        col.markdown(card_html, unsafe_allow_html=True)

    # Vade Konsensüs & Konfluans Kutusu
    c_score = horizon_res.confluence_score
    c_color = "#059669" if c_score > 0.2 else ("#DC2626" if c_score < -0.2 else "#D97706")
    confluence_html = (
        f'<div style="background-color: #0F131D; border: 1px solid #1E293B; '
        f"border-left: 4px solid {c_color}; border-radius: 8px; padding: 12px 16px; "
        f"margin-bottom: 12px; display: flex; justify-content: space-between; "
        f'align-items: center; flex-wrap: wrap; gap: 8px;">'
        f'<div><span style="font-size: 0.88rem; font-weight: 700; color: #F8FAFC;">'
        f'🏛️ Vade Konsensüsü: <span style="color: {c_color};">'
        f"{horizon_res.confluence_label}</span></span>"
        f'<div style="font-size: 0.80rem; color: #94A3B8; margin-top: 3px;">'
        f"{horizon_res.summary}</div></div>"
        f'<div style="font-size: 0.82rem; font-weight: 700; color: #94A3B8;">'
        f'Uyum Skoru: <span style="color: #F8FAFC;">{c_score:+.2f}</span></div></div>'
    )
    st.markdown(confluence_html, unsafe_allow_html=True)

    with st.expander("💡 Çoklu Vade İhtimalleri ve Fiyat Koridoru Nasıl Okunmalı?", expanded=False):
        st.markdown(
            r"""
            * **Yükseliş İhtimali (%):** Modelin ilgili vade sonunda hissenin bugünkü fiyatının
              üzerinde kapatma **olasılığıdır** (getiri oranı değil, ihtimaldir).
              Örneğin **%61.6**, hissenin 20 gün sonra kârda kapatma ihtimalini belirtir.
            * **🛡️ %95 Fiyat Koridoru:** Hissenin son 20 günlük oynaklığına (volatilite) ve
              istatistiksel Gauss dağılımına ($\pm 1.96\sigma$) dayalı olası fiyat aralığıdır.
              Piyasa şoku olmadıkça hissenin %95 ihtimalle bu iki sınır arasında kalması beklenir.
            * **🏛️ Vade Konsensüsü:** 1 Günlük, 1 Haftalık ve 1 Aylık vadelerin ortak sinyal yönünü
              sentezler (örneğin hepsi pozitifse *Güçlü Boğa*, kısa vade düşüş uzun vade yükselişse
              *Dip Alım Fırsatı*).
            """
        )


def render_financial_tools_section(
    selected_ticker: str,
    full_df: pd.DataFrame,
    current_price: float,
) -> None:
    """ADX trend gücü, Golden Cross, Pivot ve Risk/Ödül analizini render eder."""
    pivots = multi_horizon.calculate_pivot_levels(full_df)
    trend = multi_horizon.calculate_trend_strength(full_df)
    s1 = pivots.get("s1", current_price * 0.97)
    r1 = pivots.get("r1", current_price * 1.03)
    rr = multi_horizon.calculate_risk_reward(current_price, s1, r1)

    with st.expander("📐 Profesyonel Finansal Göstergeler & Risk Masası", expanded=False):
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("**📊 Trend Gücü & İvme:**")
            st.write(f"• **ADX (14):** `{trend['adx']}` ({trend['trend_strength']})")
            if trend["golden_cross"]:
                st.write("• **Golden Cross:** 🟡 **Aktif (SMA 50 > SMA 200)**")
            elif trend["death_cross"]:
                st.write("• **Death Cross:** 🔴 **Aktif (SMA 50 < SMA 200)**")
            else:
                st.write("• **Trend Kesişimi:** Nötr / Yeterli veri yok")
        with c2:
            st.markdown("**🎯 Destek & Direnç Seviyeleri (TL):**")
            p_val = pivots.get("pivot", current_price)
            r2_val = pivots.get("r2", current_price * 1.05)
            s2_val = pivots.get("s2", current_price * 0.95)
            st.write(f"• **Direnç 2 (R2):** `{r2_val:.2f}` | **R1:** `{r1:.2f}`")
            st.write(f"• **Pivot (P):** `{p_val:.2f}`")
            st.write(f"• **Destek 1 (S1):** `{s1:.2f}` | **S2:** `{s2_val:.2f}`")
        with c3:
            st.markdown("**⚖️ Risk / Ödül Oranı (R:R):**")
            st.write(f"• **R:R Oranı:** **{rr['label']}**")
            st.write(f"• **Potansiyel Kâr (R1):** `+{rr['potential_gain_tl']:.2f} TL`")
            st.write(f"• **Potansiyel Risk (S1):** `-{rr['potential_risk_tl']:.2f} TL`")

        fib_382 = pivots.get("fib_382")
        fib_500 = pivots.get("fib_500")
        fib_618 = pivots.get("fib_618")
        if fib_382 and fib_500 and fib_618:
            st.caption(
                f"Fibonacci Seviyeleri: %38.2 = **{fib_382:.2f} TL** · "
                f"%50.0 (Denge) = **{fib_500:.2f} TL** · %61.8 (Altın Oran) = **{fib_618:.2f} TL**"
            )

        exits = patterns.calculate_atr_exit_levels(full_df)
        if exits is not None:
            st.markdown("---")
            st.markdown("**🎯 Dinamik ATR Tabanlı Çıkış ve Hedef Seviyeleri (14G ATR):**")
            sl_col, tp1_col, tp2_col = st.columns(3)
            with sl_col:
                st.markdown(
                    f"🛑 **Zarar Kes (Stop-Loss):** `{exits.stop_loss:.2f} TL` "
                    f"(<span style='color:#EF4444;'>%{exits.stop_loss_pct:.1f}</span>)<br>"
                    f"<small style='color:#94A3B8;'>1.5 × ATR Koruma Seviyesi</small>",
                    unsafe_allow_html=True,
                )
            with tp1_col:
                st.markdown(
                    f"🎯 **Kâr Al 1 (Hedef 1):** `{exits.take_profit_1:.2f} TL` "
                    f"(<span style='color:#10B981;'>+%{exits.take_profit_1_pct:.1f}</span>)<br>"
                    f"<small style='color:#94A3B8;'>2.0 × ATR İlk Kâr Realizasyonu</small>",
                    unsafe_allow_html=True,
                )
            with tp2_col:
                st.markdown(
                    f"🚀 **Kâr Al 2 (Hedef 2):** `{exits.take_profit_2:.2f} TL` "
                    f"(<span style='color:#34D399;'>+%{exits.take_profit_2_pct:.1f}</span>)<br>"
                    f"<small style='color:#94A3B8;'>3.0 × ATR Trend Sürme Hedefi</small>",
                    unsafe_allow_html=True,
                )


def render_patterns_and_sr_section(
    selected_ticker: str,
    full_df: pd.DataFrame,
) -> None:
    """Klasik Japon mum formasyonları ve dinamik kırılım analizini render eder."""
    sr = patterns.calculate_support_resistance(full_df)
    c_patterns = patterns.detect_candlestick_patterns(full_df, lookback=5)

    with st.expander("🕯️ Mum Formasyonları & Destek / Direnç Kırılım Radarı", expanded=True):
        if sr is not None:
            c1, c2, c3, c4 = st.columns(4)
            c1.markdown(f"**Kırılım / Durum:**<br>{sr.status_label}", unsafe_allow_html=True)
            c2.markdown(
                f"**Direnç (R1):**<br>`{sr.resistance_1:.2f} TL` "
                f"({sr.distance_to_resistance_pct:+.1f}%)",
                unsafe_allow_html=True,
            )
            c3.markdown(f"**Pivot Seviyesi:**<br>`{sr.pivot:.2f} TL`", unsafe_allow_html=True)
            c4.markdown(
                f"**Destek (S1):**<br>`{sr.support_1:.2f} TL` (-{sr.distance_to_support_pct:.1f}%)",
                unsafe_allow_html=True,
            )

        if c_patterns:
            st.markdown("##### 🔍 Son Günlerde Tespit Edilen Mum Formasyonları:")
            for p in c_patterns[-4:]:
                st.markdown(
                    f"* **{p.label}** ({p.date}): {p.description} "
                    f"*(Sinyal Gücü: %{p.strength * 100:.0f})*"
                )
        else:
            st.caption(
                "ℹ️ Son 5 işlem gününde belirgin bir dönüş formasyonu "
                "(Çekiç, Boğa/Ayı Yutan vb.) tespit edilmedi; fiyat konsolidasyon bandında."
            )


def render_paper_trading_simulator_section(
    selected_ticker: str,
    full_df: pd.DataFrame,
    model,
    cost: float = 0.001,
) -> None:
    """TL bazlı sanal portföy ve kâğıt üzerinde alım-satım simülatörü (#52)."""
    st.subheader(
        "💼 Sanal Portföy & Kâğıt Üzerinde İşlem Simülatörü (Paper Trading)",
        anchor=False,
    )
    st.markdown(
        f"**{selected_ticker}** için belirleyeceğiniz sanal sermaye ile yapay zeka modelinin "
        f"ürettiği olasılık sinyalleri doğrultusunda işlem yapılsaydı portföyünüzün nasıl "
        f"büyüyeceğini ve 'Al ve Tut' stratejisine göre ne kadar **Alfa (Ekstra Getiri)** "
        f"sağlandığını simüle edin."
    )

    c_cap, c_thr = st.columns(2)
    with c_cap:
        init_cap = st.number_input(
            "Başlangıç Sermayesi (TL)",
            min_value=10_000,
            max_value=5_000_000,
            value=100_000,
            step=10_000,
            key=f"sim_cap_input_{selected_ticker}",
            help="Sanal portföy başlangıç nakit bakiyesi.",
        )
    with c_thr:
        sim_thr = st.slider(
            "İşleme Giriş Olasılığı Eşiği",
            min_value=0.50,
            max_value=0.70,
            value=0.53,
            step=0.01,
            key=f"sim_thresh_slider_{selected_ticker}",
            help="Model güveni bu değerin üzerine çıktığında AL, altına düştüğünde NAKDE GEÇ.",
        )

    window = backtest.recent_window(full_df, months=BACKTEST_MONTHS)
    if len(window) < 10:
        st.caption("Yeterli geçmiş veri bulunamadığı için simülasyon hesaplanamadı.")
        return

    probs = model.predict_proba(features.align_model_features(model, window))[:, 1]
    sim_res = simulator.simulate_investment(
        frame=window[["Date", "close"]],
        probs=probs,
        threshold=sim_thr,
        initial_capital=float(init_cap),
        cost=cost,
    )

    if sim_res is None:
        st.warning("Simülasyon hesaplanamadı.")
        return

    m1, m2, m3, m4 = st.columns(4)
    m1.markdown(
        f"**Son Portföy Değeri:**<br>`{sim_res.final_value:,.2f} TL` "
        f"(%{sim_res.total_return_pct:+.1f})",
        unsafe_allow_html=True,
    )
    m2.markdown(
        f"**Net Kâr / Zarar:**<br>`{sim_res.net_profit_tl:+,.2f} TL`",
        unsafe_allow_html=True,
    )
    m3.markdown(
        f"**Al ve Tut (Buy & Hold):**<br>`{sim_res.buy_hold_final_value:,.2f} TL` "
        f"(Alfa: `{sim_res.alpha_tl:+,.2f} TL`)",
        unsafe_allow_html=True,
    )
    m4.markdown(
        f"**Kazanma Oranı (Win Rate):**<br>`%{sim_res.win_rate * 100:.1f}` "
        f"({sim_res.winning_trades} Kâr / {sim_res.trades_count} İşlem)",
        unsafe_allow_html=True,
    )

    st.caption(
        f"Ödenen Komisyon: **{sim_res.total_commission_paid:,.2f} TL** · "
        f"Maksimum Çekilme (Max Drawdown): **-%{sim_res.max_drawdown_pct:.1f}** · "
        f"Zararlı İşlemler: **{sim_res.losing_trades}**"
    )

    if not sim_res.daily_timeline.empty:
        fig_sim = go.Figure()
        fig_sim.add_trace(
            go.Scatter(
                x=sim_res.daily_timeline["Date"],
                y=sim_res.daily_timeline["portfolio_value"],
                mode="lines",
                name="✨ Yapay Zeka Portföyü",
                line=dict(color="#10B981", width=2.5),
            )
        )
        fig_sim.add_trace(
            go.Scatter(
                x=sim_res.daily_timeline["Date"],
                y=sim_res.daily_timeline["buy_hold_value"],
                mode="lines",
                name="📈 Al ve Tut (Buy & Hold)",
                line=dict(color="#64748B", width=1.5, dash="dash"),
            )
        )
        fig_sim.update_layout(
            yaxis_title="Portföy Değeri (TL)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            margin=dict(l=20, r=20, t=30, b=20),
            dragmode="pan",
            hovermode="x unified",
        )
        st.plotly_chart(fig_sim, width="stretch", config=PLOTLY_CONFIG)


def render_single_ticker_tab(
    selected_ticker: str,
    selected_ticker_full: str,
    prob_low: float,
    prob_high: float,
    cost: float = 0.0,
    active_view: str = "📈 Teknik Model & Grafikler",
    *args,
    **kwargs,
):
    """Tek hisse senedi detay analizi sekmesini oluşturur."""
    v = active_view.lower()
    is_rag = "bilanço" in v or "colab" in v or "kurumsal" in v or "rag" in v or "chatbot" in v
    is_kap = "kap" in v
    is_paper = "paper" in v or "simülatör" in v
    is_tracking = (
        "audit" in v or "doğrulama" in v or "forward" in v or "takip" in v or "başarı" in v
    )
    is_drift = "drift" in v or "sapma" in v or "psi" in v
    is_technical = not (is_rag or is_kap or is_paper or is_tracking or is_drift)

    model = load_model()
    with st.spinner(f"{selected_ticker} verileri analiz ediliyor..."):
        df_processed, full_df = get_prediction_data(selected_ticker_full)
        # Sadece en son günü al (Yarın için tahmin yapacağız)
        input_data = df_processed.iloc[[-1]]

        # Gerekli Featurelar (Modele dinamik hizalanır)
        X_pred = features.align_model_features(model, input_data)

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
            st.write(f"🎯 **Modelin Seans Tahmini ({session_info.next_session_str}):**")
            signal_box(signal_label)

        with col3:
            st.write("📊 **Güven Skoru:**")
            signal_box(f"%{prob * 100:.1f} Olasılıkla Yükseliş")

        # Canlı (0 gecikmeli) kotasyon kontrolü
        live_quotes = get_cached_live_market_quotes()
        norm_sym = selected_ticker.replace(".IS", "")
        live_q = live_quotes.get(norm_sym)
        if live_q is not None:
            q_info = (
                f"🔴 **Canlı Seans Fiyatı (0 Gecikme):** `{live_q.last_price:.2f} TL` "
                f"({live_q.badge}) &nbsp;|&nbsp; 📊 **Gün İçi:** `{live_q.low:.2f} — "
                f"{live_q.high:.2f} TL` &nbsp;|&nbsp; ⚡ **Hacim:** `{live_q.volume:,.0f}`"
            )
            st.caption(q_info)

        # Hibrit Karar Motoru (Teknik ML + RAG Temel Analiz Sentezi)
        report = get_cached_fundamental_report(selected_ticker)
        f_score = report.sentiment_score if report is not None else None
        decision = hybrid_decision.evaluate_hybrid_decision(
            ticker=selected_ticker,
            technical_prob=prob,
            fundamental_score=f_score,
            prob_high=prob_high,
            prob_low=prob_low,
        )
        render_hybrid_decision_card(decision)

        # Günlük Tahmin Kaydı (Forward Tracking Denetim Defteri) (#53)
        last_date_str = str(input_data["Date"].iloc[-1]).split()[0]
        tracker.record_prediction(
            date=last_date_str,
            ticker=selected_ticker,
            signal=1 if prob >= 0.50 else 0,
            prob=prob,
            entry_price=current_price,
        )

        # Her menü öğesi kendine özel tam sayfa olarak render edilir (açılır/kapanır kutu olmadan)
        if is_rag:
            render_fundamental_rag_section(selected_ticker, report=report)
        elif is_kap:
            render_kap_llm_section(selected_ticker, report=report)
        elif is_paper:
            render_paper_trading_simulator_section(
                selected_ticker=selected_ticker,
                full_df=df_processed,
                model=model,
                cost=cost,
            )
        elif is_tracking:
            render_forward_tracking_section(selected_ticker, full_df)
        elif is_drift:
            render_drift_section(df_processed[config.FEATURES])
        elif is_technical:
            # Çoklu Vade Trend Matrisi (1 Gün • 1 Hafta • 1 Ay)
            render_multi_horizon_matrix(
                selected_ticker=selected_ticker,
                input_data=input_data,
                full_df=full_df,
                current_price=current_price,
                prob_low=prob_low,
                prob_high=prob_high,
            )

            # Profesyonel Finansal Göstergeler & Risk Masası (ADX, Pivotlar, R:R)
            render_financial_tools_section(
                selected_ticker=selected_ticker,
                full_df=full_df,
                current_price=current_price,
            )
            render_patterns_and_sr_section(selected_ticker, full_df)

            # Makro ve Piyasa Göstergeleri (#10)
            macro_cols = ["xu100_ret", "rel_strength_bist", "usdtry_change"]
            if all(col in input_data.columns for col in macro_cols):
                with st.expander(
                    "🌐 Makro Piyasa Göstergeleri (BIST 100 & USD/TRY)", expanded=False
                ):
                    m1, m2, m3 = st.columns(3)
                    xu_pct = input_data["xu100_ret"].iloc[-1] * 100
                    rel_pct = input_data["rel_strength_bist"].iloc[-1] * 100
                    fx_pct = input_data["usdtry_change"].iloc[-1] * 100
                    m1.write(f"**🏛️ BIST 100 Getirisi:** %{xu_pct:+.2f}")
                    m2.write(f"**💪 BIST 100 Rölatif Güç:** %{rel_pct:+.2f}")
                    m3.write(f"**💵 USD/TRY Değişimi:** %{fx_pct:+.2f}")

            # Sermaye Tahsisi & Pozisyon Boyutlandırma (Kelly) (#71)
            if len(full_df) >= 10:
                d_vol = float(full_df["close"].pct_change().tail(20).std())
                if np.isnan(d_vol) or d_vol <= 0:
                    d_vol = 0.02
            else:
                d_vol = 0.02
            annual_vol = float(np.clip(d_vol * np.sqrt(252.0), 0.10, 0.80))
            render_risk_sizing_section(selected_ticker, prob, annual_vol)

            # GRAFİK KISMI (Candlestick & Destek/Direnç Seviyeleri)
            st.subheader(f"{selected_ticker} - Fiyat Grafiği & Kırılım Seviyeleri", anchor=False)
            sr = patterns.calculate_support_resistance(full_df)
            fig = go.Figure(
                data=[
                    go.Candlestick(
                        x=full_df["Date"][-90:],
                        open=full_df["open"][-90:],
                        high=full_df["high"][-90:],
                        low=full_df["low"][-90:],
                        close=full_df["close"][-90:],
                        name="Fiyat",
                    )
                ]
            )
            if sr is not None and len(full_df) >= 2:
                x_dates = full_df["Date"][-90:]
                x_start = x_dates.iloc[0]
                x_end = x_dates.iloc[-1]
                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_end],
                        y=[sr.resistance_1, sr.resistance_1],
                        mode="lines",
                        line=dict(color="#EF4444", width=1.5, dash="dash"),
                        name=f"Direnç (R1): {sr.resistance_1:.2f} TL",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_end],
                        y=[sr.pivot, sr.pivot],
                        mode="lines",
                        line=dict(color="#38BDF8", width=1, dash="dot"),
                        name=f"Pivot: {sr.pivot:.2f} TL",
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=[x_start, x_end],
                        y=[sr.support_1, sr.support_1],
                        mode="lines",
                        line=dict(color="#10B981", width=1.5, dash="dash"),
                        name=f"Destek (S1): {sr.support_1:.2f} TL",
                    )
                )

                exits = patterns.calculate_atr_exit_levels(full_df)
                if exits is not None:
                    fig.add_trace(
                        go.Scatter(
                            x=[x_start, x_end],
                            y=[exits.stop_loss, exits.stop_loss],
                            mode="lines",
                            line=dict(color="#DC2626", width=1.5, dash="dot"),
                            name=(
                                f"Stop-Loss: {exits.stop_loss:.2f} TL (%{exits.stop_loss_pct:.1f})"
                            ),
                        )
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=[x_start, x_end],
                            y=[exits.take_profit_1, exits.take_profit_1],
                            mode="lines",
                            line=dict(color="#059669", width=1.5, dash="dot"),
                            name=(
                                f"Hedef 1: {exits.take_profit_1:.2f} TL "
                                f"(+%{exits.take_profit_1_pct:.1f})"
                            ),
                        )
                    )
            fig.update_layout(
                xaxis_rangeslider_visible=False,
                dragmode="pan",
                hovermode="x unified",
            )
            st.plotly_chart(fig, width="stretch", config=PLOTLY_CONFIG)

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
                dragmode=False,
                xaxis=dict(fixedrange=True),
                yaxis=dict(fixedrange=True),
            )
            st.plotly_chart(fig_explain, width="stretch", config=PLOTLY_CONFIG)
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
