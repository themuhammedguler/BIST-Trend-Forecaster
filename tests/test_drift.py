# tests/test_drift.py
"""Data Drift ve PSI (Population Stability Index) izleme modülü birim testleri (#45)."""

import numpy as np
import pandas as pd
import pytest

from src import drift


def test_calculate_psi_identical_distributions():
    """Bire bir aynı verilerde PSI 0 olmalıdır."""
    data = np.random.RandomState(42).normal(0, 1, 500)
    psi = drift.calculate_psi(data, data)
    assert psi == pytest.approx(0.0, abs=1e-6)


def test_calculate_psi_close_distributions():
    """Aynı dağılımdan gelen iki örneklemde PSI < 0.10 (Sağlıklı) olmalıdır."""
    rng = np.random.RandomState(42)
    ref = rng.normal(10, 2, 2000)
    curr = rng.normal(10, 2, 2000)

    psi = drift.calculate_psi(ref, curr)
    assert psi < 0.10

    status, badge, action = drift.classify_drift(psi)
    assert status == "Sağlıklı"
    assert "🟢" in badge


def test_calculate_psi_shifted_distributions():
    """Belirgin şekilde kaymış (drift etmiş) dağılımda PSI > 0.25 (Kritik Sapma) olmalıdır."""
    rng = np.random.RandomState(42)
    ref = rng.normal(0, 1, 1000)
    curr = rng.normal(2.5, 1, 1000)

    psi = drift.calculate_psi(ref, curr)
    assert psi > 0.25

    status, badge, action = drift.classify_drift(psi)
    assert status == "Kritik Sapma"
    assert "🔴" in badge
    assert "yeniden eğitilmesi" in action.lower()


def test_calculate_psi_edge_cases():
    """Boş, NaN, sonsuz ve sabit değerli diziler istisna üretmemelidir."""
    # Boş girişler
    assert drift.calculate_psi([], []) == 0.0
    assert drift.calculate_psi([1.0, 2.0], []) == 0.0
    assert drift.calculate_psi([], [1.0, 2.0]) == 0.0

    # NaN ve sonsuz değerler
    ref_nan = [np.nan, np.inf, -np.inf]
    curr_nan = [np.nan, 1.0]
    assert drift.calculate_psi(ref_nan, curr_nan) == 0.0

    # Sabit (sıfır varyans) değerler: İkisi de aynı
    ref_const = np.full(50, 42.0)
    curr_const = np.full(50, 42.0)
    assert drift.calculate_psi(ref_const, curr_const) == 0.0

    # Sabit değerler: Güncel farklı
    curr_diff = np.full(50, 99.0)
    assert drift.calculate_psi(ref_const, curr_diff) == 1.0


def test_classify_drift_thresholds():
    """PSI eşiklerinin (0.10 ve 0.25) sınır durumları doğru sınıflandırılmalıdır."""
    assert drift.classify_drift(0.0)[0] == "Sağlıklı"
    assert drift.classify_drift(0.0999)[0] == "Sağlıklı"
    assert drift.classify_drift(0.10)[0] == "Hafif Sapma"
    assert drift.classify_drift(0.25)[0] == "Hafif Sapma"
    assert drift.classify_drift(0.2501)[0] == "Kritik Sapma"
    assert drift.classify_drift(1.5)[0] == "Kritik Sapma"


def test_compute_feature_drift_and_summary():
    """Çoklu öznitelik tablosu ve genel sağlık özeti doğru üretilmeli ve sıralanmalıdır."""
    rng = np.random.RandomState(42)
    n = 500

    ref_df = pd.DataFrame(
        {
            "rsi": rng.uniform(30, 70, n),
            "macd": rng.normal(0, 1, n),
            "volatility": rng.normal(0.02, 0.005, n),
        }
    )

    # curr_df: rsi aynı, macd hafif kaymış, volatility kritik kaymış
    curr_df = pd.DataFrame(
        {
            "rsi": rng.uniform(30, 70, n),
            "macd": rng.normal(0.5, 1, n),
            "volatility": rng.normal(0.10, 0.005, n),
        }
    )

    drift_df = drift.compute_feature_drift(ref_df, curr_df, features=["rsi", "macd", "volatility"])

    assert len(drift_df) == 3
    assert list(drift_df.columns) == ["Öznitelik", "PSI", "Durum", "Rozet", "Öneri"]

    # En yüksek PSI en üstte olmalı (azalan sıralı)
    assert drift_df.iloc[0]["Öznitelik"] == "volatility"
    assert drift_df.iloc[0]["Durum"] == "Kritik Sapma"

    summary = drift.summarize_drift(drift_df)
    assert summary["overall_status"] == "Kritik Sapma"
    assert "🔴" in summary["badge"]
    assert "volatility" in summary["critical_features"]
    assert summary["total_features"] == 3
    assert summary["max_psi"] == pytest.approx(drift_df["PSI"].max(), abs=1e-4)


def test_compute_series_drift_by_split():
    """Zaman serisi bölme (oran ve güncel satır sayısı) doğru çalışmalıdır."""
    rng = np.random.RandomState(42)
    df = pd.DataFrame(
        {
            "rsi": rng.uniform(30, 70, 100),
            "macd": rng.normal(0, 1, 100),
        }
    )

    # Oran ile bölme
    drift_df, summary = drift.compute_series_drift_by_split(df, split_ratio=0.7)
    assert len(drift_df) == 2
    assert summary["total_features"] == 2

    # recent_rows ile bölme
    drift_df2, summary2 = drift.compute_series_drift_by_split(df, recent_rows=30)
    assert len(drift_df2) == 2
    assert summary2["total_features"] == 2

    # Çok küçük DataFrame (< 10 satır) boş tablo dönmeli
    small_df = pd.DataFrame({"rsi": [50.0, 52.0]})
    drift_small, summary_small = drift.compute_series_drift_by_split(small_df)
    assert drift_small.empty
    assert summary_small["overall_status"] == "Sağlıklı"


def test_cli_main_execution(tmp_path, monkeypatch, capsys):
    """drift.py CLI komut satırı arayüzü hatasız çalışmalıdır."""
    # Test için mini CSV verisi üret
    n = 100
    dates = pd.bdate_range("2024-01-01", periods=n)
    df = pd.DataFrame(
        {
            "Date": dates.strftime("%Y-%m-%d"),
            "close": np.linspace(10, 20, n),
            "high": np.linspace(10.5, 20.5, n),
            "low": np.linspace(9.5, 19.5, n),
            "open": np.linspace(10, 20, n),
            "volume": np.full(n, 10000),
            "ticker": ["AKBNK"] * n,
        }
    )
    test_csv = tmp_path / "test_data.csv"
    df.to_csv(test_csv, index=False)

    monkeypatch.setattr(
        "sys.argv",
        ["drift.py", "--data-path", str(test_csv), "--recent-days", "20"],
    )
    drift.main()

    captured = capsys.readouterr()
    assert "ÖZNİTELİK SAPMASI RAPORU" in captured.out
    assert "tamamlandı" in captured.out


def test_cyclical_features_classified_as_calendar_cycle():
    """month ve day_of_week gibi takvim öznitelikleri kritik sapma alarmı üretmemelidir."""
    status, badge, action = drift.classify_drift(5.0, feature_name="month")
    assert status == "Takvim Döngüsü"
    assert "📅" in badge
    assert "yeniden eğitilmesini gerektirmez" in action

    status_dow, badge_dow, _ = drift.classify_drift(2.0, feature_name="day_of_week")
    assert status_dow == "Takvim Döngüsü"
    assert "📅" in badge_dow

    # summary içinde critical_features listesine girmemeli
    drift_df = pd.DataFrame(
        [
            {
                "Öznitelik": "month",
                "PSI": 5.0,
                "Durum": "Takvim Döngüsü",
                "Rozet": "📅 Takvim",
                "Öneri": "Takvim döngüsü",
            },
            {
                "Öznitelik": "rsi",
                "PSI": 0.05,
                "Durum": "Sağlıklı",
                "Rozet": "🟢 Sağlıklı",
                "Öneri": "Sağlıklı",
            },
        ]
    )
    summary = drift.summarize_drift(drift_df)
    assert summary["overall_status"] == "Sağlıklı"
    assert "month" not in summary["critical_features"]
    assert "month" in summary["cyclical_features"]
