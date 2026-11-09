"""Gizlilik dostu ziyaretçi analitiği ve oturum takip modülü.

GoatCounter (çerezsiz, KVKK/GDPR uyumlu) ve dahili oturum sayacı ile
uygulama trafiğini, kullanıcı etkileşimini ve sayfa görüntülenmelerini izler.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from src import config

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_SESSION_COUNTER = 0


def record_session() -> int:
    """Yeni bir Streamlit oturumu başladığında sayacı bir artırır."""
    global _SESSION_COUNTER
    try:
        import streamlit as st

        if "analytics_session_recorded" not in st.session_state:
            st.session_state["analytics_session_recorded"] = True
            with _lock:
                _SESSION_COUNTER += 1
    except Exception as exc:
        logger.debug("Oturum kaydedilirken Streamlit bağlamı dışındaydı: %s", exc)
        with _lock:
            _SESSION_COUNTER += 1

    return _SESSION_COUNTER


def get_session_count() -> int:
    """Mevcut çalışma sürecindeki toplam oturum sayısını döndürür."""
    with _lock:
        return _SESSION_COUNTER


def reset_session_counter_for_tests() -> None:
    """Birim testler için oturum sayacını ve session_state durumunu sıfırlar."""
    global _SESSION_COUNTER
    with _lock:
        _SESSION_COUNTER = 0
    try:
        import streamlit as st

        if "analytics_session_recorded" in st.session_state:
            del st.session_state["analytics_session_recorded"]
    except Exception:
        pass


def build_goatcounter_html(counter_code: str) -> str:
    """GoatCounter izleme kodunu HTML script etiketi olarak döndürür."""
    clean_code = counter_code.strip()
    return (
        f'<script data-goatcounter="https://{clean_code}.goatcounter.com/count" '
        f'async src="//gc.zgo.at/count.js"></script>'
    )


def build_ga_html(tracking_id: str) -> str:
    """Google Analytics (gtag.js) izleme kodunu döndürür."""
    clean_id = tracking_id.strip()
    return f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={clean_id}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{clean_id}');
    </script>
    """


def inject_tracking_script() -> bool:
    """Yapılandırılmışsa GoatCounter veya Google Analytics izleme betiğini sayfaya enjekte eder."""
    try:
        import streamlit.components.v1 as components
    except ImportError:
        return False

    injected = False
    if config.GOATCOUNTER_CODE:
        html_code = build_goatcounter_html(config.GOATCOUNTER_CODE)
        components.html(html_code, height=0, width=0)
        injected = True
    elif config.GA_TRACKING_ID:
        html_code = build_ga_html(config.GA_TRACKING_ID)
        components.html(html_code, height=0, width=0)
        injected = True

    return injected


def get_analytics_summary() -> dict[str, Any]:
    """Mevcut analitik durumunun özetini döndürür."""
    provider = "Dahili Sayaç"
    dashboard_url = None

    if config.GOATCOUNTER_CODE:
        provider = "GoatCounter (Bulut)"
        dashboard_url = f"https://{config.GOATCOUNTER_CODE}.goatcounter.com"
    elif config.GA_TRACKING_ID:
        provider = "Google Analytics"
        dashboard_url = "https://analytics.google.com"

    return {
        "provider": provider,
        "session_count": get_session_count(),
        "is_cloud_active": bool(config.GOATCOUNTER_CODE or config.GA_TRACKING_ID),
        "dashboard_url": dashboard_url,
    }


def render_analytics_status() -> None:
    """Streamlit yan menüsünde analitik ve ziyaretçi durumunu gösterir."""
    try:
        import streamlit as st
    except ImportError:
        return

    summary = get_analytics_summary()
    if summary["is_cloud_active"] and summary["dashboard_url"]:
        st.sidebar.caption(
            f"👁️ **Ziyaretçi Analitiği:** Aktif · "
            f"[{summary['provider']}]({summary['dashboard_url']})"
        )
    else:
        st.sidebar.caption(
            f"👁️ **Oturum Sayacı:** {summary['session_count']} ziyaret · Bulut Analitiği: Hazır"
        )
