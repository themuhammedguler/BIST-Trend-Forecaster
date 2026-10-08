# tests/test_model_train.py - model_train.py ve get_temporal_split birim testleri
import numpy as np
import pandas as pd
import pytest

from model_train import get_temporal_split


@pytest.fixture
def multi_ticker_dataset():
    """3 hisse senedi, 50 iş günü, bilinen tarihler ve özellikler."""
    dates = pd.date_range("2024-01-01", periods=50, freq="B")
    records = []
    for ticker in ["AKBNK", "GARAN", "THYAO"]:
        for d in dates:
            records.append(
                {
                    "ticker": ticker,
                    "Date": d,
                    "f1": np.random.randn(),
                    "f2": np.random.randn(),
                    "target": int(np.random.rand() > 0.5),
                }
            )
    df = pd.DataFrame(records)
    # Veriyi ticker ardından Date şeklinde sırala (features.py gibi)
    df = df.sort_values(by=["ticker", "Date"]).reset_index(drop=True)
    return df


def test_temporal_split_no_lookahead_leakage(multi_ticker_dataset):
    features_list = ["f1", "f2"]
    X_train, X_test, y_train, y_test, dates_train, cutoff_date = get_temporal_split(
        multi_ticker_dataset, features_list, train_ratio=0.8
    )

    test_dates = multi_ticker_dataset.loc[multi_ticker_dataset["Date"] >= cutoff_date, "Date"]

    # 1. En kritik kural: Eğitim kümesindeki en son tarih, test kümesindeki ilk tarihten önce olmalı
    assert dates_train.max() < test_dates.min()
    assert dates_train.max() < cutoff_date
    assert test_dates.min() >= cutoff_date

    # 2. Boyutlar eşleşmeli
    assert len(X_train) == len(y_train) == len(dates_train)
    assert len(X_test) == len(y_test) == len(test_dates)
    assert len(X_train) + len(X_test) == len(multi_ticker_dataset)


def test_temporal_split_includes_all_tickers(multi_ticker_dataset):
    """Eski satır bazlı split alfabetik son hisseleri teste atıyordu.
    Tarih bazlı splitte her hissenin hem geçmişi hem geleceği olmalı."""
    features_list = ["f1", "f2"]

    X_train, X_test, _, _, _, cutoff_date = get_temporal_split(
        multi_ticker_dataset, features_list, train_ratio=0.8
    )

    train_tickers = multi_ticker_dataset.loc[
        multi_ticker_dataset["Date"] < cutoff_date, "ticker"
    ].unique()
    test_tickers = multi_ticker_dataset.loc[
        multi_ticker_dataset["Date"] >= cutoff_date, "ticker"
    ].unique()

    assert set(train_tickers) == {"AKBNK", "GARAN", "THYAO"}
    assert set(test_tickers) == {"AKBNK", "GARAN", "THYAO"}


def test_temporal_split_ratio(multi_ticker_dataset):
    features_list = ["f1", "f2"]
    total_dates = multi_ticker_dataset["Date"].nunique()  # 50

    _, _, _, _, dates_train, cutoff_date = get_temporal_split(
        multi_ticker_dataset, features_list, train_ratio=0.7
    )

    train_dates_count = dates_train.nunique()
    expected_train_dates = int(total_dates * 0.7)  # 35
    assert train_dates_count == expected_train_dates


def test_temporal_split_resets_indices(multi_ticker_dataset):
    features_list = ["f1", "f2"]
    X_train, X_test, y_train, y_test, dates_train, _ = get_temporal_split(
        multi_ticker_dataset, features_list, train_ratio=0.8
    )

    # İndekslerin 0'dan len-1'e kesintisiz gitmesi iloc ve Optuna için zorunludur
    assert list(X_train.index) == list(range(len(X_train)))
    assert list(X_test.index) == list(range(len(X_test)))
    assert list(y_train.index) == list(range(len(y_train)))
    assert list(y_test.index) == list(range(len(y_test)))
    assert list(dates_train.index) == list(range(len(dates_train)))


def test_temporal_split_insufficient_dates():
    single_date_df = pd.DataFrame(
        {
            "Date": [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-01")],
            "f1": [1.0, 2.0],
            "target": [0, 1],
        }
    )
    with pytest.raises(ValueError, match="en az 2 farklı tarih"):
        get_temporal_split(single_date_df, ["f1"])


def test_temporal_split_extreme_ratios(multi_ticker_dataset):
    features_list = ["f1", "f2"]

    # Çok küçük train_ratio: En az 1 tarih eğitimde, gerisi testte kalmalı
    X_tr_low, X_te_low, _, _, _, _ = get_temporal_split(
        multi_ticker_dataset, features_list, train_ratio=0.0001
    )
    assert len(X_tr_low) > 0
    assert len(X_te_low) > 0

    # Çok büyük train_ratio: En az 1 tarih testte kalmalı
    X_tr_high, X_te_high, _, _, _, _ = get_temporal_split(
        multi_ticker_dataset, features_list, train_ratio=0.9999
    )
    assert len(X_tr_high) > 0
    assert len(X_te_high) > 0
