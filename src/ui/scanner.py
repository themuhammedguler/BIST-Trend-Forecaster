"""BIST 30 Piyasa Fırsat Radarı ve sektör tarayıcı arayüz modülü."""

import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import (
    config,
    daily_updater,
    email_alerts,
    portfolio_optimizer,
    risk_sizing,
    scanner,
)
from src.ui.components import SCAN_COLUMN_CONFIG, get_cached_market_scan


def render_market_scanner_tab():
    """BIST 30 piyasa tarama sekmesini ve liderlik tablolarını render eder."""
    st.subheader("📊 BIST 30 Piyasa Fırsat Radarı", anchor=False)
    st.markdown("""
    Bu modül BIST 30 endeksindeki tüm hisseleri yapay zeka modelinden geçirerek
    yarın için en yüksek yükseliş potansiyeline ve düşüş riskine sahip hisseleri sıralar.
    """)

    c_btn1, c_btn2 = st.columns([3, 7])
    with c_btn1:
        if st.button("🚀 BIST 30 Taramasını Başlat", key="btn_start_scan"):
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
            sectors = sorted(scan_df["Sektör"].dropna().unique())
            selected_sectors = st.multiselect(
                "Sektöre Göre Filtrele",
                options=sectors,
                default=[],
                help="Belirli bir veya birden fazla sektördeki hisseleri listelemek için seçin.",
            )

            filtered_scan_df = (
                scan_df[scan_df["Sektör"].isin(selected_sectors)] if selected_sectors else scan_df
            )

            top_bull, top_bear = scanner.top_and_bottom(filtered_scan_df, n=5)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 🟢 En Yüksek Yükseliş Potansiyeli (Top 5)")
                st.dataframe(
                    top_bull[
                        [
                            "Hisse",
                            "Sektör",
                            "Son Fiyat (TL)",
                            "Günlük Değişim (%)",
                            "Yükseliş Olasılığı (%)",
                        ]
                    ],
                    hide_index=True,
                    column_config=SCAN_COLUMN_CONFIG,
                )
            with c2:
                st.markdown("### 🔴 Düşüş Riski En Yüksek (Top 5)")
                st.dataframe(
                    top_bear[
                        [
                            "Hisse",
                            "Sektör",
                            "Son Fiyat (TL)",
                            "Günlük Değişim (%)",
                            "Yükseliş Olasılığı (%)",
                        ]
                    ],
                    hide_index=True,
                    column_config=SCAN_COLUMN_CONFIG,
                )

            # Sektörel Ağaç Haritası (Treemap)
            with st.expander("🗺️ BIST 30 Sektörel Görünüm & Isı Haritası (Treemap)", expanded=True):
                st.markdown(
                    "Sektörlerin ve hisselerin yapay zeka modeline göre toplu görünümü "
                    "(Yeşil: Yükseliş Beklentisi / Kırmızı: Zayıf veya Düşüş Beklentisi):"
                )
                c_t1, _ = st.columns([6, 4])
                with c_t1:
                    color_mode = st.radio(
                        "Isı Haritası Renklendirme Ölçütü:",
                        options=[
                            "Yapay Zeka Yükseliş Olasılığı (%)",
                            "Günlük Fiyat Değişimi (%)",
                        ],
                        horizontal=True,
                        key="treemap_color_mode",
                    )
                color_key = "daily_change" if "Günlük" in color_mode else "ai_prob"
                fig_treemap = portfolio_optimizer.build_sector_treemap_figure(
                    filtered_scan_df, color_by=color_key
                )
                st.plotly_chart(fig_treemap, use_container_width=True)

            with st.expander(
                "💼 Önerilen Portföy Sermaye Dağılımı (Markowitz MPT & Kelly)", expanded=False
            ):
                st.markdown(
                    "Yükseliş potansiyeli en yüksek hisseler arasında modern portföy teorisi "
                    "ve dinamik sermaye bütçelemesine göre optimize edilmiş dağılım (#71):"
                )
                tab_markowitz, tab_kelly = st.tabs(
                    ["📊 Markowitz (Maksimum Sharpe)", "🛡️ Kelly Kriteri"]
                )

                with tab_markowitz:
                    st.caption(
                        "En yüksek Sharpe oranını (risk başına düşen getiri) hedefleyen "
                        "Markowitz Mean-Variance optimizasyonu:"
                    )
                    c_m1, c_m2 = st.columns(2)
                    with c_m1:
                        target_capital = st.number_input(
                            "Yatırım Sermayesi (TL)",
                            min_value=1_000.0,
                            max_value=10_000_000.0,
                            value=100_000.0,
                            step=10_000.0,
                            key="mpt_capital",
                        )
                    with c_m2:
                        max_w = st.slider(
                            "Hisse Başına Azami Ağırlık (%)",
                            min_value=20,
                            max_value=50,
                            value=35,
                            step=5,
                            key="mpt_max_w",
                        )

                    mpt_res = portfolio_optimizer.compute_markowitz_allocation(
                        filtered_scan_df,
                        max_tickers=5,
                        total_capital=float(target_capital),
                        max_weight=float(max_w) / 100.0,
                    )

                    if not mpt_res.allocation_table.empty:
                        exp_pct = mpt_res.expected_annual_return * 100.0
                        vol_pct = mpt_res.annual_volatility * 100.0
                        st.markdown(
                            f"📈 **Beklenen Yıllık Getiri:** `%{exp_pct:.1f}` "
                            f"&nbsp;|&nbsp; ⚡ **Yıllık Volatilite:** `%{vol_pct:.1f}` "
                            f"&nbsp;|&nbsp; 🎯 **Sharpe Oranı:** `{mpt_res.sharpe_ratio:.2f}`"
                        )
                        c_chart, c_table = st.columns([4, 6])
                        with c_chart:
                            fig_donut = portfolio_optimizer.build_portfolio_allocation_figure(
                                mpt_res.allocation_table
                            )
                            st.plotly_chart(fig_donut, use_container_width=True)
                        with c_table:
                            st.markdown(
                                portfolio_optimizer.format_markowitz_as_markdown(
                                    mpt_res.allocation_table
                                )
                            )
                    else:
                        st.caption("Yeterli yükseliş sinyali veren hisse bulunamadı.")

                with tab_kelly:
                    alloc_df = risk_sizing.calculate_portfolio_allocation(
                        filtered_scan_df, max_tickers=5
                    )
                    if not alloc_df.empty:
                        md_table = risk_sizing.format_allocation_as_markdown(alloc_df)
                        st.markdown(md_table)
                    else:
                        st.caption("Eşik üzeri yükseliş sinyali veren hisse bulunamadı.")

            # Otonom Seans Kapanış E-Posta Alarm Robotu (#71, #80)
            render_email_alert_section(filtered_scan_df)

            st.markdown("### 📋 Tüm BIST 30 Liderlik Tablosu")
            st.dataframe(
                filtered_scan_df[
                    [
                        "Hisse",
                        "Sektör",
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


def render_email_alert_section(filtered_scan_df: pd.DataFrame) -> None:
    """Otonom seans kapanış e-posta bülteni test ve önizleme bileşeni."""
    with st.expander("📬 Otonom Seans Kapanış E-Posta Alarm Robotu", expanded=False):
        st.markdown(
            """
            **BIST Trend Forecaster Otonom Alarm Robotu**, her iş günü BIST seansı kapandıktan sonra
            (**18:30 TRT / 15:30 UTC**) GitHub Actions üzerinde otomatik olarak devreye girer.

            * 🌲 **Çoklu Vade Taraması:** 1 Günlük, 1 Haftalık ve 1 Aylık modelleri çalıştırır.
            * 🏛️ **Vade Konsensüsü:** *GÜÇLÜ BOĞA* ve *DİP ALIM FIRSATI* veren hisseleri filtreler.
            * 🛡️ **Risk & Fiyat Koridoru:** %95 fiyat koridorunu ve Kelly sermaye payını hesaplar.
            * ✉️ **Yönetici Bülteni:** Derin Obsidyen temalı raporu e-posta adresinize ulaştırır.
            """
        )

        c_prev, c_send = st.columns([1, 1], gap="medium")

        with c_prev:
            st.markdown("#### 👁️ Canlı Bülten Önizleme")
            st.caption(
                "Bugünün piyasa tarama verileriyle hazırlanan e-posta bülteninin tam görünümü:"
            )
            if st.button("📄 Bugünün Bültenini Önizle", key="btn_preview_newsletter"):
                st.session_state["show_newsletter_preview"] = True

        with c_send:
            st.markdown("#### 📨 Test E-Postası Gönder")
            recipient_input = st.text_input(
                "Alıcı E-Posta Adresi",
                value=os.getenv("ALERT_RECIPIENT_EMAIL", ""),
                placeholder="ornek@alanadi.com",
                key="input_alert_email",
                help="Bültenin ulaştırılacağı e-posta adresini girin.",
            )
            if st.button("🚀 Test Gönderimi Yap", key="btn_send_test_email"):
                if not recipient_input:
                    st.warning("Lütfen geçerli bir alıcı e-posta adresi giriniz.")
                else:
                    html_content, picks = daily_updater.build_daily_newsletter(filtered_scan_df)
                    if not picks:
                        st.warning("Bültene eklenebilecek yükseliş sinyalli hisse bulunamadı.")
                    else:
                        with st.spinner("E-posta gönderiliyor..."):
                            success = email_alerts.send_email_alert(
                                subject="🏛️ [TEST] BIST 30 Seans Kapanış ve Çoklu Vade Bülteni",
                                html_content=html_content,
                                recipient_email=recipient_input,
                            )
                        if success:
                            st.success(f"✅ Test e-postası başarıyla iletildi: {recipient_input}")
                        else:
                            st.error(
                                "❌ E-posta gönderilemedi! Sunucu veya SMTP yapılandırması eksik."
                            )
                            with st.expander("🛠️ 2 Dakikada Kolay SMTP Kurulum Rehberi"):
                                st.markdown(
                                    """
                                    **Gmail ile Otomatik Gönderim Adımları:**
                                    1. Google Hesabınıza girin &rarr; **Güvenlik** &rarr;
                                       **2 Adımlı Doğrulama**'yı açın.
                                    2. Sayfanın en altındaki **Uygulama Şifreleri (App Passwords)**
                                       bölümüne gidin.
                                    3. *"BIST Forecaster"* adında yeni şifre üretin (16 haneli kod).
                                    4. GitHub deponuzda **Settings &rarr; Secrets and variables
                                       &rarr; Actions** menüsüne şu secret'ları ekleyin:
                                       * `SMTP_SERVER`: `smtp.gmail.com`
                                       * `SMTP_PORT`: `587`
                                       * `SMTP_USER`: `epostaniz@gmail.com`
                                       * `SMTP_PASSWORD`: *(16 haneli uygulama şifreniz)*
                                       * `ALERT_RECIPIENT_EMAIL`: *(Raporun gideceği adres)*
                                    """
                                )

        if st.session_state.get("show_newsletter_preview", False):
            html_preview, _ = daily_updater.build_daily_newsletter(filtered_scan_df)
            st.markdown("---")
            st.markdown("##### 📱 Bülten Önizleme Penceresi (HTML Responsive):")
            components.html(html_preview, height=620, scrolling=True)
