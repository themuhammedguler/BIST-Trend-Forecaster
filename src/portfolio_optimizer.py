"""Sektörel Ağaç Haritası (Treemap) & Markowitz Portföy Optimizasyonu Modülü.

Modern Portföy Teorisi (MPT / Mean-Variance Optimization) ile hisselerin
tahmin edilen getirilerini ve kovaryans matrisini birleştirerek en yüksek
Sharpe oranını (risk başına düşen getiri) veren optimal sermaye dağılımını hesaplar.
Ayrıca BIST 30 hisselerinin sektörel yapay zeka görünümünü interaktif ağaç haritası ile sunar.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from scipy.optimize import minimize

from src import config

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MarkowitzResult:
    """Markowitz portföy optimizasyonu sonuç veri yapısı."""

    weights: dict[str, float]
    expected_annual_return: float
    annual_volatility: float
    sharpe_ratio: float
    allocation_table: pd.DataFrame


def optimize_sharpe_portfolio(
    tickers: list[str],
    expected_returns: np.ndarray,
    cov_matrix: np.ndarray,
    risk_free_rate: float = 0.40,  # BIST için risksiz faiz oranı (yıllık %40)
    max_weight: float = 0.35,  # Tek bir hisseye verilebilecek azami sınır
    total_capital: float = 100_000.0,
    scan_df: pd.DataFrame | None = None,
) -> MarkowitzResult:
    """En yüksek Sharpe oranına sahip optimal portföy ağırlıklarını hesaplar."""
    n = len(tickers)
    if n == 0 or len(expected_returns) != n or cov_matrix.shape != (n, n):
        empty_table = pd.DataFrame(
            columns=["Hisse", "Sektör", "Ağırlık (%)", "Tutar (TL)", "Yükseliş Olasılığı (%)"]
        )
        return MarkowitzResult({}, 0.0, 0.0, 0.0, empty_table)

    # Amaç fonksiyonu: Negatif Sharpe Oranını minimize et
    def objective(w: np.ndarray) -> float:
        port_return = np.sum(w * expected_returns)
        port_vol = np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))
        if port_vol <= 1e-6:
            return 1e6
        sharpe = (port_return - risk_free_rate) / port_vol
        return -sharpe

    # Kısıtlar: Ağırlıklar toplamı = 1.0 (Tam yatırım)
    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1.0}]

    # Sınırlar: 0.0 <= w_i <= max_weight (Kaldıraçsız ve açığa satışsız)
    actual_max = max(1.0 / n, min(1.0, max_weight))
    bounds = tuple((0.0, actual_max) for _ in range(n))

    # Başlangıç noktası: Eşit ağırlık
    initial_weights = np.ones(n) / n

    res = minimize(
        objective,
        initial_weights,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if res.success:
        raw_weights = res.x
    else:
        # Yakınsama sağlanamazsa eşit ağırlık kullan
        raw_weights = initial_weights

    # Küçük küsuratları temizle (< %1 ise sıfırla ve yeniden normalize et)
    cleaned = np.where(raw_weights < 0.01, 0.0, raw_weights)
    tot = np.sum(cleaned)
    final_weights = cleaned / tot if tot > 0 else initial_weights

    weights_dict = {tickers[i]: round(float(final_weights[i]), 4) for i in range(n)}

    exp_ret = float(np.sum(final_weights * expected_returns))
    vol = float(np.sqrt(np.dot(final_weights.T, np.dot(cov_matrix, final_weights))))
    sharpe = float((exp_ret - risk_free_rate) / vol) if vol > 1e-6 else 0.0

    # Tabloyu oluştur
    records = []
    prob_map = {}
    if scan_df is not None and not scan_df.empty:
        for _, r in scan_df.iterrows():
            t_clean = str(r.get("Hisse", "")).split()[0]
            prob_map[t_clean] = float(r.get("Yükseliş Olasılığı (%)", 50.0))

    for i, t in enumerate(tickers):
        w = final_weights[i]
        if w >= 0.01:
            clean_ticker = t.replace(".IS", "")
            sector = config.get_ticker_sector(clean_ticker)
            alloc_tl = round(w * total_capital, 2)
            records.append(
                {
                    "Hisse": clean_ticker,
                    "Sektör": sector,
                    "Ağırlık (%)": round(w * 100.0, 1),
                    "Tutar (TL)": alloc_tl,
                    "Yükseliş Olasılığı (%)": prob_map.get(clean_ticker, 50.0),
                }
            )

    table_df = pd.DataFrame(records)
    if not table_df.empty:
        table_df = table_df.sort_values(by="Ağırlık (%)", ascending=False).reset_index(drop=True)

    return MarkowitzResult(
        weights=weights_dict,
        expected_annual_return=round(exp_ret, 4),
        annual_volatility=round(vol, 4),
        sharpe_ratio=round(sharpe, 2),
        allocation_table=table_df,
    )


def format_markowitz_as_markdown(allocation_df: pd.DataFrame) -> str:
    """Markowitz tahsisat tablosunu şık bir markdown tablosuna dönüştürür."""
    if allocation_df is None or allocation_df.empty:
        return ""
    lines = [
        "| Hisse | Sektör | Portföy Payı | Alım Tutarı | AI Yükseliş Güveni |",
        "| :--- | :--- | :---: | :---: | :---: |",
    ]
    for _, row in allocation_df.iterrows():
        hisse = row.get("Hisse", "")
        sektor = row.get("Sektör", "")
        agirlik = row.get("Ağırlık (%)", 0.0)
        tutar = row.get("Tutar (TL)", 0.0)
        prob = row.get("Yükseliş Olasılığı (%)", 50.0)
        lines.append(f"| **{hisse}** | {sektor} | %{agirlik:.1f} | {tutar:,.2f} TL | %{prob:.1f} |")
    return "\n".join(lines)


def build_sector_treemap_figure(
    scan_df: pd.DataFrame,
    color_by: str = "ai_prob",
) -> go.Figure:
    """BIST 30 hisselerinin sektörel yapay zeka görünümünü Plotly Treemap olarak üretir."""
    if scan_df is None or scan_df.empty:
        return go.Figure()

    df = scan_df.copy()

    # Hiyerarşi veri listeleri
    labels: list[str] = ["BIST 30"]
    parents: list[str] = [""]
    values: list[float] = [0.0]
    color_values: list[float] = [50.0]
    hover_texts: list[str] = ["Tüm BIST 30 Hisseleri"]

    # 1. Sektörleri topla
    sector_groups = df.groupby("Sektör")
    sector_means = {}

    for sector, group in sector_groups:
        if color_by == "daily_change":
            val = float(group["Günlük Değişim (%)"].mean())
        else:
            val = float(group["Yükseliş Olasılığı (%)"].mean())
        sector_means[sector] = val

        labels.append(str(sector))
        parents.append("BIST 30")
        values.append(len(group))
        color_values.append(val)
        hover_texts.append(
            f"<b>Sektör: {sector}</b><br>Hisse: {len(group)} adet<br>AI Güveni: %{val:.1f}"
        )

    # 2. Hisseleri ekle
    for _, row in df.iterrows():
        hisse = str(row.get("Hisse", ""))
        clean_name = hisse.split()[0]
        sector = str(row.get("Sektör", "Diğer"))
        price = float(row.get("Son Fiyat (TL)", 0.0))
        prob = float(row.get("Yükseliş Olasılığı (%)", 50.0))
        change = float(row.get("Günlük Değişim (%)", 0.0))
        signal = str(row.get("Tahmin", ""))

        labels.append(clean_name)
        parents.append(sector)
        values.append(1.0)

        c_val = change if color_by == "daily_change" else prob
        color_values.append(c_val)

        hover_texts.append(
            f"<b>{clean_name}</b> ({sector})<br>"
            f"Fiyat: {price:.2f} TL<br>"
            f"AI Yükseliş Güveni: %{prob:.1f}<br>"
            f"Günlük Değişim: %{change:+.2f}<br>"
            f"Sinyal: {signal}"
        )

    # Renk ölçeği: Kırmızı -> Gri/Sarı -> Yeşil
    if color_by == "daily_change":
        c_min, c_max, c_mid = -5.0, 5.0, 0.0
        c_title = "Günlük Değişim (%)"
    else:
        c_min, c_max, c_mid = 35.0, 65.0, 50.0
        c_title = "Yükseliş Olasılığı (%)"

    fig = go.Figure(
        go.Treemap(
            labels=labels,
            parents=parents,
            values=values,
            branchvalues="remainder",
            marker=dict(
                colors=color_values,
                colorscale="RdYlGn",
                cmid=c_mid,
                cmin=c_min,
                cmax=c_max,
                colorbar=dict(title=c_title),
                showscale=True,
            ),
            hoverinfo="text",
            hovertext=hover_texts,
            texttemplate="<b>%{label}</b>",
            textposition="middle center",
            pathbar=dict(visible=True),
        )
    )

    fig.update_layout(
        margin=dict(t=30, l=10, r=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=520,
    )
    return fig


def get_historical_covariance_matrix(tickers: list[str]) -> pd.DataFrame:
    """BIST 30 hisselerinin yıllıklandırılmış kovaryans matrisini hesaplar."""
    try:
        if os.path.exists(config.DATA_PATH):
            df = pd.read_csv(config.DATA_PATH)
            piv = df.pivot(index="Date", columns="ticker", values="close").pct_change(
                fill_method=None
            )
            clean_tickers = [t.replace(".IS", "") for t in tickers]
            available = [t for t in clean_tickers if t in piv.columns]
            if available:
                sub_cov = piv[available].cov() * 252.0
                return sub_cov
    except Exception as exc:
        logger.debug("Kovaryans matrisi hesaplanırken hata oluştu: %s", exc)

    # Yedek: Korelasyonsuz varsayılan volatilite matrisi
    n = len(tickers)
    return pd.DataFrame(np.eye(n) * 0.09, index=tickers, columns=tickers)


def compute_markowitz_allocation(
    scan_df: pd.DataFrame,
    max_tickers: int = 5,
    total_capital: float = 100_000.0,
    max_weight: float = 0.35,
    risk_free_rate: float = 0.40,
) -> MarkowitzResult:
    """Piyasa taramasındaki en iyi hisseler için Markowitz MPT portföyünü optimize eder."""
    if scan_df is None or scan_df.empty:
        empty_table = pd.DataFrame(
            columns=["Hisse", "Sektör", "Ağırlık (%)", "Tutar (TL)", "Yükseliş Olasılığı (%)"]
        )
        return MarkowitzResult({}, 0.0, 0.0, 0.0, empty_table)

    # Sadece pozitif yükseliş potansiyeline sahip hisseleri al
    bullish = scan_df[scan_df["_prob"] >= 0.50].sort_values(by="_prob", ascending=False)
    if bullish.empty:
        bullish = scan_df.sort_values(by="_prob", ascending=False)

    top_candidates = bullish.head(max_tickers)
    tickers = [str(r["Hisse"]).split()[0] for _, r in top_candidates.iterrows()]
    probs = np.array([float(r["_prob"]) for _, r in top_candidates.iterrows()])

    # Beklenen getiri: Risksiz faiz oranı + yapay zeka model avantajı
    expected_returns = risk_free_rate + (probs - 0.50) * 1.5

    cov_df = get_historical_covariance_matrix(tickers)
    valid_tickers = [t for t in tickers if t in cov_df.columns]
    if len(valid_tickers) < len(tickers):
        cov_matrix = np.eye(len(tickers)) * 0.09
    else:
        cov_matrix = cov_df.loc[tickers, tickers].values

    return optimize_sharpe_portfolio(
        tickers=tickers,
        expected_returns=expected_returns,
        cov_matrix=cov_matrix,
        risk_free_rate=risk_free_rate,
        max_weight=max_weight,
        total_capital=total_capital,
        scan_df=top_candidates,
    )


def build_portfolio_allocation_figure(
    allocation_df: pd.DataFrame,
) -> go.Figure:
    """Optimal portföy dağılımını halka (donut) grafiği olarak çizer."""
    if allocation_df is None or allocation_df.empty:
        return go.Figure()

    fig = go.Figure(
        go.Pie(
            labels=allocation_df["Hisse"],
            values=allocation_df["Ağırlık (%)"],
            hole=0.45,
            textinfo="label+percent",
            hoverinfo="label+value+percent",
            marker=dict(line=dict(color="#0F172A", width=2)),
        )
    )

    fig.update_layout(
        margin=dict(t=20, l=10, r=10, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=380,
        showlegend=True,
    )
    return fig
