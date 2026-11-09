"""Ziyaretçi analitiği ve oturum takibi modülü (src/analytics.py) birim testleri."""

from src import analytics, config


def test_session_counter():
    analytics.reset_session_counter_for_tests()
    assert analytics.get_session_count() == 0

    count = analytics.record_session()
    assert count == 1
    assert analytics.get_session_count() == 1


def test_build_goatcounter_html():
    html = analytics.build_goatcounter_html("my-bist-app")
    assert 'data-goatcounter="https://my-bist-app.goatcounter.com/count"' in html
    assert "//gc.zgo.at/count.js" in html
    assert "allow_frame: true" in html
    assert (
        '<img src="https://my-bist-app.goatcounter.com/count?p=/&t=BIST_Trend_Forecaster"' in html
    )


def test_send_server_side_beacon(monkeypatch):
    called_url = []

    class DummyResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def dummy_urlopen(req, timeout=3):
        called_url.append(req.full_url)
        return DummyResponse()

    import urllib.request

    monkeypatch.setattr(urllib.request, "urlopen", dummy_urlopen)
    analytics._send_server_side_beacon("demo-code")
    assert len(called_url) == 1
    assert "https://demo-code.goatcounter.com/count?p=/&t=BIST_Trend_Forecaster" in called_url[0]


def test_send_server_side_beacon_handles_exception(monkeypatch):
    import urllib.request

    def failing_urlopen(req, timeout=3):
        raise OSError("Network unreachable")

    monkeypatch.setattr(urllib.request, "urlopen", failing_urlopen)
    # Should not raise exception
    analytics._send_server_side_beacon("demo-code")


def test_build_ga_html():
    html = analytics.build_ga_html("G-XXXXXXXXXX")
    assert "https://www.googletagmanager.com/gtag/js?id=G-XXXXXXXXXX" in html
    assert "gtag('config', 'G-XXXXXXXXXX')" in html


def test_analytics_summary_default(monkeypatch):
    monkeypatch.setattr(config, "GOATCOUNTER_CODE", "")
    monkeypatch.setattr(config, "GA_TRACKING_ID", "")

    summary = analytics.get_analytics_summary()
    assert summary["is_cloud_active"] is False
    assert summary["provider"] == "Dahili Sayaç"
    assert summary["dashboard_url"] is None


def test_analytics_summary_goatcounter(monkeypatch):
    monkeypatch.setattr(config, "GOATCOUNTER_CODE", "bist-trend")
    monkeypatch.setattr(config, "GA_TRACKING_ID", "")

    summary = analytics.get_analytics_summary()
    assert summary["is_cloud_active"] is True
    assert "GoatCounter" in summary["provider"]
    assert summary["dashboard_url"] == "https://bist-trend.goatcounter.com"


def test_inject_tracking_script_when_disabled(monkeypatch):
    monkeypatch.setattr(config, "GOATCOUNTER_CODE", "")
    monkeypatch.setattr(config, "GA_TRACKING_ID", "")

    injected = analytics.inject_tracking_script()
    assert injected is False
