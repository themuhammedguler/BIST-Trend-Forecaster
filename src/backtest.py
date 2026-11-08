# src/backtest.py - model sinyalleriyle basit al/nakit stratejisinin geçmiş performansı
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import risk_sizing

TRADING_DAYS_PER_YEAR = 252


def strategy_returns(close, positions, cost=0.0):
    """
    Günlük strateji ve al-tut getirilerini hesaplar.

    positions[t], t günü kapanışında alınan pozisyondur (1: hissede, 0: nakitte)
    ve t -> t+1 kapanışları arasındaki getiriyi kazanır. Getiriler gerçekleştikleri
    güne (t+1) yazılır; son günün pozisyonu henüz sonuçlanmadığından kullanılmaz.
    Strateji nakitte başlar; her pozisyon değişikliğinde `cost` oranında maliyet düşülür.
    """
    close = pd.Series(close, dtype=float)
    positions = pd.Series(np.asarray(positions, dtype=float), index=close.index)

    asset = close.pct_change()
    held = positions.shift(1)
    trades = positions.diff().abs()
    trades.iloc[0] = abs(positions.iloc[0])  # nakitten ilk giriş
    strategy = held * asset - cost * trades.shift(1)

    return pd.DataFrame({"strategy": strategy, "buy_hold": asset}).iloc[1:]


def cumulative_returns(returns):
    """Bileşik kümülatif getiri: (1 + r).cumprod() - 1."""
    return (1 + returns).cumprod() - 1


def max_drawdown(returns):
    """Başlangıç sermayesi dahil, en yüksek seviyeden en büyük düşüş (negatif oran)."""
    equity = (1 + returns).cumprod()
    peak = equity.cummax().clip(lower=1.0)
    return float(min((equity / peak - 1).min(), 0.0))


def sharpe_ratio(returns, periods_per_year=TRADING_DAYS_PER_YEAR):
    """Yıllıklandırılmış Sharpe oranı (risksiz faiz 0). Oynaklık yoksa tanımsızdır (NaN)."""
    std = returns.std(ddof=1)
    if not std > 0:
        return float("nan")
    return float(returns.mean() / std * np.sqrt(periods_per_year))


def win_rate(returns):
    """İşlem yapılan (sıfırdan farklı) günlerde pozitif getiri oranı. Aktif gün yoksa 0.0."""
    returns = pd.Series(returns, dtype=float).dropna()
    active = returns[returns != 0.0]
    if len(active) == 0:
        return 0.0
    return float((active > 0).mean())


def profit_factor(returns):
    """Toplam kazanç / toplam kayıp oranı (gains / abs(losses)). Kayıp yoksa inf döner."""
    returns = pd.Series(returns, dtype=float).dropna()
    gains = float(returns[returns > 0].sum())
    losses = float(abs(returns[returns < 0].sum()))
    if losses == 0.0:
        return float("inf") if gains > 0.0 else 0.0
    return float(gains / losses)


def summarize(returns):
    """Toplam getiri, Sharpe oranı, maksimum değer kaybı, kazanma oranı ve kâr faktörü."""
    if len(returns) == 0:
        return {
            "total_return": 0.0,
            "sharpe": float("nan"),
            "max_drawdown": 0.0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
        }
    return {
        "total_return": float(cumulative_returns(returns).iloc[-1]),
        "sharpe": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
        "win_rate": win_rate(returns),
        "profit_factor": profit_factor(returns),
    }


@dataclass
class BacktestResult:
    returns: pd.DataFrame  # gerçekleşen günlük getiriler (strategy, buy_hold)
    curve: pd.DataFrame  # ilk günden itibaren kümülatif getiri eğrileri (0'dan başlar)
    strategy: dict
    buy_hold: dict
    exposure: float  # hissede geçirilen gün oranı
    trades: int  # pozisyon değişikliği sayısı (nakitten ilk giriş dahil)
    cost: float = 0.0  # işlem başına komisyon ve kayma oranı
    sizing: str = "binary"  # "binary" veya "kelly" (#71)
    mean_position_size: float = 0.0  # ortalama sermaye tahsis oranı (#71)


def run_backtest(
    frame,
    probs,
    threshold,
    cost=0.0,
    sizing="binary",
    fraction=0.5,
    target_vol=0.25,
):
    """
    Yükseliş olasılığı `threshold` ve üzerindeyse hissede kalır, değilse nakde geçer.
    `sizing="kelly"` durumunda dinamik pozisyon boyutlandırma uygular (#71).

    frame: 'Date' ve 'close' sütunlu, tarihe göre sıralı veri; probs: her satır için
    modelin yükseliş olasılığı (o günün kapanışındaki özniteliklerden).
    """
    if len(frame) < 2:
        raise ValueError("Backtest için en az 2 işlem günü gereklidir.")

    dates = pd.DatetimeIndex(frame["Date"])
    close = pd.Series(frame["close"].to_numpy(dtype=float), index=dates)

    if sizing == "kelly":
        probs_arr = np.asarray(probs, dtype=float)
        vols = (
            frame["volatility"].to_numpy(dtype=float)
            if "volatility" in frame.columns
            else np.full(len(frame), 0.25)
        )
        pos_values = [
            risk_sizing.calculate_position_size(
                p,
                vol,
                min_threshold=threshold,
                fraction=fraction,
                target_volatility=target_vol,
            )
            for p, vol in zip(probs_arr, vols)
        ]
        positions = pd.Series(pos_values, index=dates)
    else:
        positions = pd.Series((np.asarray(probs) >= threshold).astype(float), index=dates)

    returns = strategy_returns(close, positions, cost=cost)
    start = pd.DataFrame({"strategy": [0.0], "buy_hold": [0.0]}, index=dates[:1])
    curve = pd.concat([start, cumulative_returns(returns)])

    realised = positions.iloc[:-1]
    trades = int(realised.diff().abs().fillna(realised.iloc[0]).gt(0.001).sum())

    return BacktestResult(
        returns=returns,
        curve=curve,
        strategy=summarize(returns["strategy"]),
        buy_hold=summarize(returns["buy_hold"]),
        exposure=float((realised > 0.0).mean()),
        trades=trades,
        cost=cost,
        sizing=sizing,
        mean_position_size=float(realised.mean()),
    )


def recent_window(frame, months=6):
    """Son `months` aylık dönemi döndürür (son tarihe göre)."""
    start = frame["Date"].max() - pd.DateOffset(months=months)
    return frame[frame["Date"] > start].reset_index(drop=True)
