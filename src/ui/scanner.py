"""BIST 30 Piyasa Fırsat Radarı ve sektör tarayıcı arayüz modülü."""

import streamlit as st

from src import config, risk_sizing, scanner
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

            with st.expander(
                "💼 Önerilen Portföy Sermaye Dağılımı (Kelly Criterion)", expanded=False
            ):
                st.markdown(
                    "Yükseliş potansiyeli en yüksek hisseler arasında Kelly Kriteri ve oynaklık "
                    "bütçelemesine göre optimize edilmiş sermaye tahsisi (#71):"
                )
                alloc_df = risk_sizing.calculate_portfolio_allocation(
                    filtered_scan_df, max_tickers=5
                )
                if not alloc_df.empty:
                    md_table = risk_sizing.format_allocation_as_markdown(alloc_df)
                    st.markdown(md_table)
                else:
                    st.caption("Eşik üzeri yükseliş sinyali veren hisse bulunamadı.")

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
