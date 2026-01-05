import os
import time
from pathlib import Path
from typing import Optional

from curl_cffi import requests as cffi_requests  # yfinance expects curl_cffi sessions
import pandas as pd
import yfinance as yf


CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _create_session_with_custom_ca():
    """Create a curl_cffi session that respects CA env vars and optional insecure flag."""
    session = cffi_requests.Session()
    session.headers.update(
        {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )
        }
    )
    ca_bundle = os.getenv("REQUESTS_CA_BUNDLE") or os.getenv("CURL_CA_BUNDLE")
    if ca_bundle:
        session.verify = ca_bundle
    if os.getenv("ALLOW_INSECURE_SSL") == "1":
        session.verify = False  # unsafe, only for temporary development use
    return session


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    safe = ticker.replace("/", "_").upper()
    safe_period = period.replace("/", "_")
    safe_interval = interval.replace("/", "_")
    return CACHE_DIR / f"{safe}__{safe_period}__{safe_interval}.csv"


def _load_from_cache(
    ticker: str, period: str, interval: str, max_age_hours: int = 24
) -> Optional[pd.DataFrame]:
    path = _cache_path(ticker, period, interval)
    if not path.exists():
        return None

    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        if df is not None and not df.empty:
            df = df.sort_index()
        return df
    except Exception:
        return None


def _save_to_cache(ticker: str, period: str, interval: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        df.to_csv(_cache_path(ticker, period, interval))
    except Exception:
        pass


def load_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    retries: int = 3,
    backoff_seconds: int = 2,
) -> pd.DataFrame:
    """
    Load historical data for ticker with caching and retries.

    period: "1mo","3mo","6mo","1y","2y","5y","max"
    interval: "1d","1h","15m", ...
    """
    cached = _load_from_cache(ticker, period, interval)
    if cached is not None and not cached.empty:
        return cached

    session = _create_session_with_custom_ca()
    stock = yf.Ticker(ticker, session=session)

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = stock.history(period=period, interval=interval)
            if df is not None and not df.empty:
                df = df.sort_index()
                _save_to_cache(ticker, period, interval, df)
                return df
        except Exception as exc:
            last_exc = exc

        time.sleep(backoff_seconds * (2**attempt))

    # fallback: allow stale cache up to 7 days
    stale = _load_from_cache(ticker, period, interval, max_age_hours=24 * 7)
    if stale is not None and not stale.empty:
        return stale

    if last_exc:
        raise last_exc

    raise ValueError(f"Ticker '{ticker}' liefert keine Daten (und kein Cache vorhanden).")