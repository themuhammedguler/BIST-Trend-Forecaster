"""BIST 30 Piyasa Fırsat Radarı ve sektör tarayıcı arayüz modülü."""

import json
import os

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from src import (
    config,
    daily_updater,
    email_alerts,
    live_market,
    portfolio_optimizer,
    risk_sizing,
    scanner,
)
from src.ui.components import SCAN_COLUMN_CONFIG, get_cached_market_scan

PLOTLY_CONFIG: dict[str, object] = {
    "scrollZoom": True,
    "displayModeBar": "hover",
    "doubleClick": "reset+autosize",
    "responsive": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


@st.cache_data(ttl=30, show_spinner=False)
def get_cached_live_market_quotes() -> dict[str, live_market.LiveQuote]:
    """BIST 30 hisselerinin anlık canlı kotasyonlarını 30 saniye boyunca önbellekte tutar."""
    return live_market.fetch_live_quotes()


def render_market_scanner_tab():
    """BIST 30 piyasa tarama sekmesini ve liderlik tablolarını render eder."""
    st.subheader("📊 BIST 30 Piyasa Fırsat Radarı", anchor=False)
    st.markdown("""
    Bu modül BIST 30 endeksindeki tüm hisseleri yapay zeka modelinden geçirerek
    yarın için en yüksek yükseliş potansiyeline ve düşüş riskine sahip hisseleri sıralar.
    """)

    # Anlık Canlı BIST 30 Seans Akışı (0 Gecikmeli Fiyatlar)
    quotes = get_cached_live_market_quotes()
    if quotes:
        sorted_quotes = sorted(quotes.values(), key=lambda q: q.change_pct, reverse=True)
        top_gainers = [
            f"<strong>{q.ticker}</strong> {q.badge} ({q.last_price:.2f} TL)"
            for q in sorted_quotes[:4]
        ]
        top_losers = [
            f"<strong>{q.ticker}</strong> {q.badge} ({q.last_price:.2f} TL)"
            for q in sorted_quotes[-3:]
        ]
        gainers_html = " &nbsp;•&nbsp; ".join(top_gainers)
        losers_html = " &nbsp;•&nbsp; ".join(top_losers)
        box_style = (
            "background: rgba(15, 23, 42, 0.7); "
            "border: 1px solid rgba(56, 189, 248, 0.3); "
            "border-radius: 8px; padding: 10px 14px; margin-bottom: 14px; "
            "font-size: 0.86rem;"
        )
        st.markdown(
            f"""
            <div style="{box_style}">
                <div style="display: flex; justify-content: space-between; margin-bottom: 6px;">
                    <span style="color: #38BDF8; font-weight: 700;">
                        🔴 BIST CANLI PİYASA NABZI (0 GECİKME)
                    </span>
                    <span style="color: #94A3B8; font-size: 0.78rem;">
                        Anlık Son Fiyat & Değişim
                    </span>
                </div>
                <div style="color: #F1F5F9; line-height: 1.6;">
                    🚀 <strong>Günün Liderleri:</strong> {gainers_html}<br/>
                    ⚠️ <strong>En Çok Gerileyenler:</strong> {losers_html}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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

            # Sektörel Ağaç Haritası (Treemap) & Göreceli Güç
            with st.expander("🗺️ BIST 30 Sektörel Görünüm & Isı Haritası (Treemap)", expanded=True):
                st.markdown(
                    "Sektörlerin ve hisselerin yapay zeka modeline göre toplu görünümü "
                    "(Yeşil: Yükseliş Beklentisi / Kırmızı: Zayıf veya Düşüş Beklentisi):"
                )
                if (
                    "Sektör" in filtered_scan_df.columns
                    and "Yükseliş Olasılığı (%)" in filtered_scan_df.columns
                ):
                    sec_avg = (
                        filtered_scan_df.groupby("Sektör")["Yükseliş Olasılığı (%)"]
                        .mean()
                        .sort_values(ascending=False)
                    )
                    if not sec_avg.empty:
                        top_sec = sec_avg.index[0]
                        top_val = sec_avg.iloc[0]
                        st.info(
                            f"🏆 **Günün En Güçlü Sektörü: {top_sec}** "
                            f"(Ortalama Model Güveni: **%{top_val:.1f}** &nbsp;|&nbsp; "
                            f"Taranan Sektör Sayısı: **{len(sec_avg)}**)"
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
                st.plotly_chart(fig_treemap, use_container_width=True, config=PLOTLY_CONFIG)

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
                            st.plotly_chart(
                                fig_donut, use_container_width=True, config=PLOTLY_CONFIG
                            )
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

            # Kişisel İzleme Listesi & Portföy Sepetim
            render_custom_watchlist_section(filtered_scan_df)

            # Otonom Seans Kapanış E-Posta Alarm Robotu (#71, #80)
            render_email_alert_section(filtered_scan_df)

            c_head1, c_head2 = st.columns([7, 3])
            with c_head1:
                st.markdown("### 📋 Tüm BIST 30 Liderlik Tablosu")
            with c_head2:
                csv_bytes = filtered_scan_df.to_csv(index=False).encode("utf-8-sig")
                st.download_button(
                    label="📥 CSV / Excel Olarak İndir",
                    data=csv_bytes,
                    file_name=f"BIST30_Firsat_Radari_{pd.Timestamp.now().strftime('%Y%m%d')}.csv",
                    mime="text/csv",
                    key="btn_download_scan_csv",
                    help="Tüm piyasa tarama tablosunu Excel uyumlu CSV formatında indirir.",
                )

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

            # Model Mimarisi & Performans İncelemesi
            render_model_diagnostics_section()
        else:
            st.warning("Piyasa verileri taranırken veri alınamadı.")


def render_custom_watchlist_section(scan_df: pd.DataFrame) -> None:
    """Kullanıcının seçtiği hisselerden oluşan ve URL ile kalıcılaştırılan kişisel izleme sepeti."""
    with st.expander("⭐ Kişisel İzleme Listesi & Portföy Sepetim", expanded=False):
        st.markdown(
            "Takip etmek istediğiniz hisseleri seçin. Seçiminiz tarayıcı adresine "
            "(**URL parametresi**) işlenir; böylece **sayfa yenilense (F5) veya yer imlerine "
            "eklense dahi kaybolmaz**, tamamen sizin tarayıcınıza özel kalır."
        )

        all_tickers = sorted(scan_df["Hisse"].unique()) if "Hisse" in scan_df.columns else []

        raw_param = st.query_params.get("watchlist", "")
        url_defaults = [
            t.strip().upper() for t in raw_param.split(",") if t.strip().upper() in all_tickers
        ]

        selected_watchlist = st.multiselect(
            "Sepete Hisse Ekle / Çıkar:",
            options=all_tickers,
            default=url_defaults,
            key="ms_custom_watchlist",
            help="İzleme sepetinize dahil etmek istediğiniz BIST 30 hisselerini seçin.",
        )

        if selected_watchlist:
            st.query_params["watchlist"] = ",".join(selected_watchlist)
        elif "watchlist" in st.query_params:
            del st.query_params["watchlist"]

        if selected_watchlist:
            wl_df = scan_df[scan_df["Hisse"].isin(selected_watchlist)].copy()
            avg_prob = wl_df["Yükseliş Olasılığı (%)"].mean()
            avg_chg = wl_df["Günlük Değişim (%)"].mean()

            st.markdown("##### 📊 Kişisel Sepet Analizi:")
            w1, w2, w3 = st.columns(3)
            with w1:
                st.markdown(
                    f"**Ortalama Model Güveni:**<br>"
                    f"<span style='font-size:1.15rem; font-weight:800; color:#10B981;'>"
                    f"%{avg_prob:.1f}</span>",
                    unsafe_allow_html=True,
                )
            with w2:
                st.markdown(
                    f"**Günün Ortalama Getirisi:**<br>"
                    f"<span style='font-size:1.15rem; font-weight:800; color:#38BDF8;'>"
                    f"%{avg_chg:+.2f}</span>",
                    unsafe_allow_html=True,
                )
            with w3:
                badge = (
                    "🟢 Güçlü Alım"
                    if avg_prob >= 55.0
                    else ("🟡 Nötr" if avg_prob >= 48.0 else "🔴 Temkinli")
                )
                st.markdown(
                    f"**Sepet Görünümü:**<br>"
                    f"<span style='font-size:1.15rem; font-weight:800;'>{badge}</span>",
                    unsafe_allow_html=True,
                )

            st.table(
                wl_df[
                    [
                        "Hisse",
                        "Sektör",
                        "Son Fiyat (TL)",
                        "Günlük Değişim (%)",
                        "Yükseliş Olasılığı (%)",
                        "Tahmin",
                    ]
                ]
            )
        else:
            st.caption(
                "ℹ️ Henüz hisse seçilmedi. Yukarıdan hisse seçerek sepetinizi oluşturabilirsiniz."
            )


def render_model_diagnostics_section() -> None:
    """Yapay Zeka model mimarisi, doğruluk metrikleri ve denetim defteri özetini render eder."""
    with st.expander("⚙️ Yapay Zeka Model Mimarisi & Performans İncelemesi", expanded=False):
        st.markdown(
            """
            BIST Trend Forecaster, çoklu makine öğrenimi modellerinin ağırlıklı birleşimi olan
            **v2.1-ensemble** (XGBoost, LightGBM, CatBoost) mimarisiyle çalışır.
            Aşağıda test seti üzerinde ölçülen başarı metrikleri ve günlük tahmin denetim defteri
            özeti yer almaktadır:
            """
        )

        c_meta1, c_meta2 = st.columns([6, 4])
        with c_meta1:
            st.markdown("##### 🏆 Model Topluluğu Başarı Kıyaslaması (Test Seti)")
            meta_path = "models/ensemble_meta.json"
            if os.path.exists(meta_path):
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta_data = json.load(f)
                    df_meta = pd.DataFrame(meta_data)
                    st.table(df_meta)
                except Exception:
                    st.caption("Model metaveri dosyası okunamadı.")
            else:
                st.caption("ensemble_meta.json bulunamadı.")

        with c_meta2:
            st.markdown("##### 📋 Otonom Tahmin Defteri (Audit Ledger)")
            audit_path = "data/prediction_audit.json"
            if os.path.exists(audit_path):
                try:
                    with open(audit_path, "r", encoding="utf-8") as f:
                        audit_data = json.load(f)
                    total_records = len(audit_data)
                    bull_count = sum(
                        1
                        for r in audit_data
                        if r.get("signal") == 1 or "YÜKSELİŞ" in r.get("direction", "")
                    )
                    bear_count = total_records - bull_count
                    last_record = audit_data[-1] if audit_data else {}
                    last_date = last_record.get("date", "—")

                    st.markdown(
                        f"* **Toplam Denetlenen Tahmin:** `{total_records} Adet`\n"
                        f"* **Son Güncelleme:** `{last_date}`\n"
                        f"* **Sinyal Dağılımı:** 🟢 {bull_count} Alım · 🔴 {bear_count} Satış/Nötr"
                    )
                except Exception:
                    st.caption("Denetim defteri okunamadı.")
            else:
                st.caption("prediction_audit.json bulunamadı.")

        st.info(
            "🔄 **Otonom Model Yeniden Eğitimi (Retrain):** Sistem, GitHub Actions "
            "(`retrain.yml`) iş akışı üzerinden her hafta sonu yeni gelen seans verilerini "
            "otomatik olarak veri tabanına işler, hiperparametreleri optimize eder ve "
            "model ağırlıklarını günceller."
        )


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
                                    **Seçenek 1: Resend (Önerilen, Hızlı ve Güvenli)**
                                    1. [resend.com](https://resend.com) üzerinden ücretsiz hesap
                                       açın ve bir API Key alın (`re_...`).
                                    2. GitHub deponuzda **Settings &rarr; Secrets &rarr; Actions**
                                       menüsüne şu değerleri ekleyin:
                                       * `SMTP_SERVER`: `smtp.resend.com`
                                       * `SMTP_PORT`: `465` (veya `587`)
                                       * `SMTP_USER`: `resend`
                                       * `SMTP_PASSWORD`: *(Resend API Key'iniz)*
                                       * `ALERT_RECIPIENT_EMAIL`: *(Kayıtlı e-postanız)*

                                    ---
                                    **Seçenek 2: Gmail ile Gönderim Adımları**
                                    1. Google Hesabınız &rarr; **Güvenlik** &rarr;
                                       **2 Adımlı Doğrulama**'yı açın.
                                    2. Sayfanın altındaki **Uygulama Şifreleri** bölümüne gidin.
                                    3. *"BIST Forecaster"* adında yeni şifre üretin (16 hane).
                                    4. GitHub repository secrets'a ekleyin:
                                       * `SMTP_SERVER`: `smtp.gmail.com`
                                       * `SMTP_PORT`: `587`
                                       * `SMTP_USER`: `epostaniz@gmail.com`
                                       * `SMTP_PASSWORD`: *(16 haneli uygulama şifresi)*
                                       * `ALERT_RECIPIENT_EMAIL`: *(Raporun gideceği adres)*
                                    """
                                )

        if st.session_state.get("show_newsletter_preview", False):
            html_preview, _ = daily_updater.build_daily_newsletter(filtered_scan_df)
            st.markdown("---")
            st.markdown("##### 📱 Bülten Önizleme Penceresi (HTML Responsive):")
            components.html(html_preview, height=620, scrolling=True)
