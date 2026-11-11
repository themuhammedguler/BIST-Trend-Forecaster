"""Keep-Alive ping utility for Streamlit Cloud and Hugging Face Spaces.

Periodically called by GitHub Actions (.github/workflows/keep_alive.yml)
every 6 hours to prevent free-tier containers from falling asleep.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

import requests

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("keep_alive")

DEFAULT_TARGETS: list[str] = [
    # Streamlit Community Cloud (Ana sayfa ve dahili sağlık kontrolü)
    "https://bist-trend-forecaster.streamlit.app",
    "https://bist-trend-forecaster.streamlit.app/_stcore/health",
    # Hugging Face Spaces (Doğrudan Container ve Space Sayfası)
    "https://themuhammedguler-bist-trend-forecaster.hf.space",
    "https://themuhammedguler-bist-trend-forecaster.hf.space/_stcore/health",
    "https://huggingface.co/spaces/themuhammedguler/BIST-Trend-Forecaster",
]

BROWSER_HEADERS: dict[str, str] = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "tr-TR,tr;q=0.9,en-US;q=0.8,en;q=0.7",
    "Cache-Control": "no-cache",
    "Pragma": "no-cache",
}


def ping_url(
    session: requests.Session, url: str, timeout: float = 30.0
) -> tuple[bool, int, float, str]:
    """Pings a single URL using a persistent session and returns (success, code, elapsed, msg)."""
    start = time.perf_counter()
    try:
        resp = session.get(url, timeout=timeout, allow_redirects=True)
        elapsed = time.perf_counter() - start
        success = resp.status_code in (200, 301, 302, 303, 304, 307, 308)
        return success, resp.status_code, elapsed, resp.reason
    except requests.RequestException as req_err:
        elapsed = time.perf_counter() - start
        return False, 0, elapsed, str(req_err)


def run_keep_alive(targets: list[str] | None = None) -> int:
    """Executes keep-alive ping round across all target URLs."""
    urls = targets or DEFAULT_TARGETS
    logger.info("Starting keep-alive ping round for %d target URLs...", len(urls))

    session = requests.Session()
    session.headers.update(BROWSER_HEADERS)

    overall_success = True
    for idx, url in enumerate(urls, 1):
        logger.info("[%d/%d] Pinging %s", idx, len(urls), url)
        success, code, elapsed, msg = ping_url(session, url)

        if success:
            logger.info("  -> Status %d in %.2fs [%s]", code, elapsed, msg)
        else:
            logger.warning(
                "  -> Slow or Cold Container (status: %d, time: %.2fs): %s", code, elapsed, msg
            )
            logger.info("  -> Waiting 5s for cold start and retrying...")
            time.sleep(5)
            r_success, r_code, r_elapsed, r_msg = ping_url(session, url, timeout=40.0)
            if r_success:
                logger.info("  -> Retry successful! Status %d in %.2fs", r_code, r_elapsed)
            else:
                logger.warning("  -> Retry also failed: %s", r_msg)
                overall_success = False

    logger.info(
        "Keep-alive ping round completed. (Status: %s)", "SUCCESS" if overall_success else "PARTIAL"
    )
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Keep-Alive pinger for Streamlit & HF Spaces")
    parser.add_argument(
        "urls", nargs="*", help="Optional URLs to ping (defaults to configured apps)"
    )
    args = parser.parse_args()

    custom_urls = args.urls or [
        u.strip() for u in os.getenv("KEEP_ALIVE_URLS", "").split(",") if u.strip()
    ]
    sys.exit(run_keep_alive(custom_urls or None))


if __name__ == "__main__":
    main()
