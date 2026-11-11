"""Unit tests for the keep-alive ping script."""

from unittest.mock import MagicMock, patch

import requests

from scripts.keep_alive_ping import ping_url, run_keep_alive


def test_ping_url_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.reason = "OK"

    session = MagicMock(spec=requests.Session)
    session.get.return_value = mock_resp

    success, status, elapsed, msg = ping_url(session, "https://example.com")
    assert success is True
    assert status == 200
    assert elapsed >= 0.0
    assert msg == "OK"


def test_ping_url_redirect_success():
    mock_resp = MagicMock()
    mock_resp.status_code = 303
    mock_resp.reason = "See Other"

    session = MagicMock(spec=requests.Session)
    session.get.return_value = mock_resp

    success, status, elapsed, msg = ping_url(session, "https://example.com")
    assert success is True
    assert status == 303


def test_ping_url_failure_handled():
    session = MagicMock(spec=requests.Session)
    session.get.side_effect = requests.RequestException("Connection refused")

    success, status, elapsed, msg = ping_url(session, "https://invalid.local")
    assert success is False
    assert status == 0
    assert "Connection refused" in msg


def test_run_keep_alive_dry_run():
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.reason = "OK"

    with patch("requests.Session.get", return_value=mock_resp):
        code = run_keep_alive(["https://example.com/health"])
        assert code == 0
