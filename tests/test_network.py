# tests/test_network.py - Yahoo Finance istekleri için retry/backoff testleri
import pandas as pd
import pytest
import yfinance

from src import network


@pytest.fixture
def sleeps(monkeypatch):
    """Gerçek bekleme yerine istenen gecikmeleri kaydeder."""
    recorded = []
    monkeypatch.setattr(network, "_sleep", recorded.append)
    return recorded


@pytest.fixture
def frame():
    return pd.DataFrame({"Close": [10.0, 10.5]}, index=pd.bdate_range("2025-01-01", periods=2))


def scripted_download(monkeypatch, outcomes):
    """Her çağrıda sıradaki sonucu döndürür; Exception ise fırlatır."""
    calls = []

    def download(*args, **kwargs):
        calls.append((args, kwargs))
        outcome = outcomes[len(calls) - 1]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(yfinance, "download", download)
    return calls


def test_returns_immediately_on_success(monkeypatch, sleeps, frame):
    calls = scripted_download(monkeypatch, [frame])
    result = network.download_with_retry("AKBNK.IS", period="6mo", progress=False)
    assert result.equals(frame)
    assert calls == [(("AKBNK.IS",), {"period": "6mo", "progress": False})]
    assert sleeps == []


@pytest.mark.parametrize("failures", [1, 2])
def test_recovers_after_transient_errors(monkeypatch, sleeps, frame, failures):
    outcomes = [ConnectionError("ağ hatası")] * failures + [frame]
    calls = scripted_download(monkeypatch, outcomes)
    assert network.download_with_retry("AKBNK.IS").equals(frame)
    assert len(calls) == failures + 1


def test_retries_empty_responses(monkeypatch, sleeps, frame):
    # yfinance ağ/hız sınırı hatalarını yutup boş DataFrame döndürür
    calls = scripted_download(monkeypatch, [pd.DataFrame(), frame])
    assert network.download_with_retry("AKBNK.IS").equals(frame)
    assert len(calls) == 2


def test_backoff_is_exponential_and_capped(monkeypatch, sleeps):
    scripted_download(monkeypatch, [pd.DataFrame()] * 5)
    network.download_with_retry("AKBNK.IS", attempts=5, min_delay=2, max_delay=10)
    assert sleeps == [2, 4, 8, 10]


def test_default_policy_is_three_attempts(monkeypatch, sleeps):
    calls = scripted_download(monkeypatch, [pd.DataFrame()] * 3)
    network.download_with_retry("AKBNK.IS")
    assert len(calls) == 3
    assert sleeps == [2, 4]


def test_returns_last_empty_result_when_no_data(monkeypatch, sleeps):
    """Boş sonuç hata değildir (örn. geçersiz sembol); çağıran kendi mesajını üretir."""
    scripted_download(monkeypatch, [pd.DataFrame()] * 3)
    assert network.download_with_retry("INVALID.IS").empty


def test_raises_friendly_error_when_all_attempts_fail(monkeypatch, sleeps):
    scripted_download(monkeypatch, [ConnectionError("ağ hatası")] * 3)
    with pytest.raises(network.MarketDataUnavailableError, match="bağlantı sorunu") as exc:
        network.download_with_retry("AKBNK.IS")
    # Uygulama ValueError'ları kullanıcıya uyarı olarak gösterir
    assert isinstance(exc.value, ValueError)
    assert isinstance(exc.value.__cause__, ConnectionError)


def test_does_not_sleep_after_last_attempt(monkeypatch, sleeps):
    scripted_download(monkeypatch, [ConnectionError()] * 2)
    with pytest.raises(network.MarketDataUnavailableError):
        network.download_with_retry("AKBNK.IS", attempts=2)
    assert sleeps == [2]


def test_ensure_ascii_ca_bundle_noop_on_non_windows(monkeypatch):
    """Windows harici işletim sistemlerinde sertifika yönlendirmesi çalıştırılmaz."""
    monkeypatch.setattr("os.name", "posix")
    network.ensure_ascii_ca_bundle()


def test_ensure_ascii_ca_bundle_noop_when_path_is_ascii(monkeypatch):
    """Sertifika yolu zaten standart ASCII ise ortam değişkenleri değiştirilmez."""
    monkeypatch.setattr("os.name", "nt")
    monkeypatch.setattr("certifi.where", lambda: "C:/standard/path/cacert.pem")
    network.ensure_ascii_ca_bundle()
