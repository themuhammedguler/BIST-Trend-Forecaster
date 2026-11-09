"""Tek hisse senedi detay analizi, SHAP açıklamaları, grafikler ve backtest sekmesi."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import (
    backtest,
    config,
    drift,
    explain,
    features,
    fundamental_rag,
    hybrid_decision,
    kap_llm,
    kap_news,
    multi_horizon,
    risk_sizing,
    session_calendar,
    signals,
    tracker,
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


def render_fundamental_rag_section(
    selected_ticker: str, report: fundamental_rag.FundamentalReport | None = None
):
    """Financial Analyst ChatBot Project temel analiz ve bilanço özetini gösterir."""
    with st.expander("💬 Kurumsal Temel Analiz & Bilanço Chatbot'u (RAG)", expanded=False):
        if report is None:
            report = fundamental_rag.fetch_fundamental_report(selected_ticker)
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
        else:
            st.markdown(
                f"ℹ️ *{selected_ticker} için önceden üretilmiş statik özet rapor bulunamadı. "
                f"Yukarıdaki butona tıklayarak **Financial Analyst ChatBot** üzerinden "
                f"canlı RAG analizini başlatabilirsiniz.*"
            )


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
            <span>🤖 <strong>Teknik Olasılık:</strong> %{decision.technical_prob * 100:.1f}</span>
            <span>📑 <strong>KAP Temel Duygu:</strong> {fund_text}</span>
            <span>⚖️ <strong>Ağırlık:</strong> %55 Teknik + %45 Temel</span>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)


def render_forward_tracking_section(selected_ticker: str, full_df: pd.DataFrame):
    """Kayıtlı tahminlerin gerçekleşen piyasa fiyatlarıyla başarı karnesini gösterir (#53)."""
    with st.expander(
        "🎯 Gerçekleşen Model Başarısı & İleriye Dönük Takip (Forward Tracking)",
        expanded=False,
    ):
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
                wait_div = (
                    '<div style="font-size: 0.75rem; color: #64748B;">Seans tamamlanmadı</div>'
                )
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
            f'<div style="background-color: #0F131D; border: 1px solid #1A2233; '
            f"border-radius: 8px; padding: 12px 14px; margin-bottom: 10px; "
            f'border-top: 3px solid {border_color};">'
            f'<div style="font-size: 0.80rem; color: #94A3B8; font-weight: 600; '
            f'margin-bottom: 6px;">{title}</div>'
            f'<div style="display: flex; justify-content: space-between; '
            f'align-items: center; margin-bottom: 8px;">'
            f'<span style="font-size: 1.15rem; font-weight: 800; color: #F8FAFC;">'
            f"%{fc.probability * 100:.1f}</span>"
            f'<span style="background-color: {bg_badge}; color: {text_color}; '
            f"font-size: 0.75rem; font-weight: 700; padding: 2px 8px; "
            f'border-radius: 4px; border: 1px solid {border_color};">'
            f"{icon} {fc.label}</span></div>"
            f'<div style="font-size: 0.76rem; color: #64748B; border-top: 1px solid #1A2233; '
            f'padding-top: 6px;">%95 Fiyat Koridoru (±%{corr["pct"]:.1f}):<br/>'
            f'<strong style="color: #CBD5E1;">{corr["min"]:.2f} TL — '
            f"{corr['max']:.2f} TL</strong></div></div>"
        )
        col.markdown(card_html, unsafe_allow_html=True)

    # Vade Konsensüs & Konfluans Kutusu
    c_score = horizon_res.confluence_score
    c_color = "#059669" if c_score > 0.2 else ("#DC2626" if c_score < -0.2 else "#D97706")
    confluence_html = (
        f'<div style="background-color: #0F131D; border: 1px solid #1A2233; '
        f"border-left: 4px solid {c_color}; border-radius: 6px; padding: 10px 14px; "
        f"margin-bottom: 14px; display: flex; justify-content: space-between; "
        f'align-items: center; flex-wrap: wrap; gap: 8px;">'
        f'<div><span style="font-size: 0.85rem; font-weight: 700; color: #F8FAFC;">'
        f'🏛️ Vade Konsensüsü: <span style="color: {c_color};">'
        f"{horizon_res.confluence_label}</span></span>"
        f'<div style="font-size: 0.78rem; color: #94A3B8; margin-top: 2px;">'
        f"{horizon_res.summary}</div></div>"
        f'<div style="font-size: 0.80rem; font-weight: 700; color: #94A3B8;">'
        f'Uyum Skoru: <span style="color: #F8FAFC;">{c_score:+.2f}</span></div></div>'
    )
    st.markdown(confluence_html, unsafe_allow_html=True)


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
            st.write(f"🤖 **Modelin Seans Tahmini ({session_info.next_session_str}):**")
            signal_box(signal_label)

        with col3:
            st.write("📊 **Güven Skoru:**")
            signal_box(f"%{prob * 100:.1f} Olasılıkla Yükseliş")

        # Hibrit Karar Motoru (Teknik ML + RAG Temel Analiz Sentezi)
        report = fundamental_rag.fetch_fundamental_report(selected_ticker)
        f_score = report.sentiment_score if report is not None else None
        decision = hybrid_decision.evaluate_hybrid_decision(
            ticker=selected_ticker,
            technical_prob=prob,
            fundamental_score=f_score,
            prob_high=prob_high,
            prob_low=prob_low,
        )
        render_hybrid_decision_card(decision)

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

        # Günlük Tahmin Kaydı (Forward Tracking Denetim Defteri) (#53)
        last_date_str = str(input_data["Date"].iloc[-1]).split()[0]
        tracker.record_prediction(
            date=last_date_str,
            ticker=selected_ticker,
            signal=1 if prob >= 0.50 else 0,
            prob=prob,
            entry_price=current_price,
        )

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
        if len(full_df) >= 10:
            d_vol = float(full_df["close"].pct_change().tail(20).std())
            if np.isnan(d_vol) or d_vol <= 0:
                d_vol = 0.02
        else:
            d_vol = 0.02
        annual_vol = float(np.clip(d_vol * np.sqrt(252.0), 0.10, 0.80))
        render_risk_sizing_section(selected_ticker, prob, annual_vol)

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
        render_forward_tracking_section(selected_ticker, full_df)
        render_drift_section(df_processed[config.FEATURES])
        render_kap_llm_section(selected_ticker)
        render_fundamental_rag_section(selected_ticker, report=report)
