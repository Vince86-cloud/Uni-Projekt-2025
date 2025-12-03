# Data fetch with optional CA handling, insecure flag, retry, and caching
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
            # Browser-like UA to reduce 429/blocked responses
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


def _cache_path(ticker: str) -> Path:
    safe = ticker.replace("/", "_").upper()
    return CACHE_DIR / f"{safe}.csv"


def _load_from_cache(ticker: str, max_age_hours: int = 24) -> Optional[pd.DataFrame]:
    path = _cache_path(ticker)
    if not path.exists():
        return None
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None
    try:
        return pd.read_csv(path, index_col=0, parse_dates=True)
    except Exception:
        return None


def _save_to_cache(ticker: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    try:
        df.to_csv(_cache_path(ticker))
    except Exception:
        pass


def load_data(ticker: str, retries: int = 3, backoff_seconds: int = 2) -> pd.DataFrame:
    """Load 1y of historical data for the given ticker with retries and caching."""
    cached = _load_from_cache(ticker)
    if cached is not None and not cached.empty:
        return cached

    session = _create_session_with_custom_ca()
    stock = yf.Ticker(ticker, session=session)

    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            df = stock.history(period="1y")
            if df is not None and not df.empty:
                _save_to_cache(ticker, df)
                return df
        except Exception as exc:
            last_exc = exc
        time.sleep(backoff_seconds * (2**attempt))

    stale = _load_from_cache(ticker, max_age_hours=24 * 7)
    if stale is not None and not stale.empty:
        return stale

    if last_exc:
        raise last_exc
    raise RuntimeError("Failed to load data and no cache available.")
