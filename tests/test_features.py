# tests/test_features.py - features.add_features() birim testleri
import numpy as np
import pandas as pd
import pytest

from features import add_features


@pytest.fixture
def ohlcv_dataset():
    """2 hisse x 90 iş günü, sma_50/bollinger ısınma periyodunu geçecek kadar uzun."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-01-01", periods=90)
    records = []
    for ticker in ["AKBNK", "GARAN"]:
        price = 100.0
        for d in dates:
            price += rng.normal(scale=1.0)
            records.append(
                {
                    "Date": d,
                    "ticker": ticker,
                    "open": price,
                    "high": price + 1,
                    "low": price - 1,
                    "close": price,
                    "volume": rng.integers(1_000, 10_000),
                }
            )
    return pd.DataFrame(records)


def test_default_drops_most_recent_day_per_ticker(ohlcv_dataset):
    """Eğitimde olduğu gibi: next_close hesaplanamayan (bilinmeyen hedefli) son gün
    veri setinden düşürülmelidir."""
    processed = add_features(ohlcv_dataset)
    max_input_date = ohlcv_dataset["Date"].max()

    assert processed["Date"].max() < max_input_date


def test_keeping_incomplete_target_retains_most_recent_day(ohlcv_dataset):
    """Canlı tahmin senaryosu: next_close/target bilinmese de en güncel günün
    özellik sütunları (rsi, macd, ...) doludur ve atılmamalıdır."""
    max_input_date = ohlcv_dataset["Date"].max()

    processed = add_features(ohlcv_dataset, drop_incomplete_target=False)

    assert processed["Date"].max() == max_input_date

    last_rows = processed[processed["Date"] == max_input_date]
    assert len(last_rows) == ohlcv_dataset["ticker"].nunique()

    feature_cols = [
        "rsi",
        "macd",
        "sma_10",
        "sma_50",
        "bb_width",
        "volatility",
        "lag_1_ret",
        "lag_2_ret",
        "vol_change",
    ]
    assert not last_rows[feature_cols].isna().any().any()


def test_keeping_incomplete_target_does_not_change_earlier_rows(ohlcv_dataset):
    """drop_incomplete_target=False sadece hedefi bilinmeyen son günü etkilemeli,
    eğitim için kullanılan önceki satırları değiştirmemeli."""
    default_processed = add_features(ohlcv_dataset)
    kept_processed = add_features(ohlcv_dataset, drop_incomplete_target=False)

    common_dates = default_processed["Date"].max()
    pd.testing.assert_frame_equal(
        kept_processed[kept_processed["Date"] <= common_dates].reset_index(drop=True),
        default_processed.reset_index(drop=True),
    )


def test_empty_dataframe_returns_empty():
    assert add_features(pd.DataFrame()).empty


def test_single_ticker_vectorized_matches_multi_ticker_exact(ohlcv_dataset):
    """Tek hisse için çalışan hızlı yol, çoklu hisse groupby mantığıyla
    bit-seviyesinde bire bir aynı değerleri üretmelidir (#14)."""
    single = ohlcv_dataset[ohlcv_dataset["ticker"] == "AKBNK"].copy().reset_index(drop=True)

    # Tekil hisse ile çalıştırıldığında optimize yol tetiklenir
    fast_result = add_features(single, drop_incomplete_target=False).reset_index(drop=True)

    # Çoklu hisse veri seti içinde aynı AKBNK satırlarını çıkar
    multi_result = add_features(ohlcv_dataset, drop_incomplete_target=False)
    multi_akbnk = multi_result[multi_result["ticker"] == "AKBNK"].reset_index(drop=True)

    pd.testing.assert_frame_equal(fast_result, multi_akbnk)
