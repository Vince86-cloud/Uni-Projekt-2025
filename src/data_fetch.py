import os
import time
from pathlib import Path
from typing import Optional, Tuple

from curl_cffi import requests as cffi_requests  # Von yfinance benötigte HTTP-Session
import pandas as pd
import yfinance as yf

# Verzeichnis zum Zwischenspeichern (Caching) von Kursdaten als CSV-Dateien
CACHE_DIR = Path(__file__).resolve().parent.parent / "data_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _create_session_with_custom_ca():
    """Erstellt eine benutzerdefinierte HTTP-Session für yfinance.

    Eigenschaften:
    - Setzt einen realistischen User-Agent (vermeidet Blockierungen)
    - Berücksichtigt benutzerdefinierte CA-Zertifikate aus Umgebungsvariablen
    - Optionales Deaktivieren der SSL-Prüfung für Entwicklungsumgebungen
    """
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
        session.verify = False  # nicht für Produktivbetrieb empfohlen

    return session


def _cache_path(ticker: str, period: str, interval: str) -> Path:
    """
    Erzeugt einen dateisystemsicheren Pfad für eine Cache-Datei
    basierend auf Ticker, Zeitraum und Intervall.
    """
    safe = ticker.replace("/", "_").upper()
    safe_period = period.replace("/", "_")
    safe_interval = interval.replace("/", "_")
    return CACHE_DIR / f"{safe}__{safe_period}__{safe_interval}.csv"


def _load_from_cache(
    ticker: str, period: str, interval: str, max_age_hours: int = 24
) -> Optional[pd.DataFrame]:
    """
    Lädt Kursdaten aus dem Cache, sofern:
    - die Datei existiert
    - sie nicht älter als max_age_hours ist

    Gibt None zurück, wenn kein gültiger Cache vorhanden ist.
    """
    path = _cache_path(ticker, period, interval)
    if not path.exists():
        return None

    # Alter der Cache-Datei in Stunden berechnen
    age_hours = (time.time() - path.stat().st_mtime) / 3600
    if age_hours > max_age_hours:
        return None

    try:
        df = pd.read_csv(path, index_col=0, parse_dates=True)

        if df is None or df.empty:
            return None

        # Index bereinigen (wichtig für Forecast und Zeitreihenoperationen)
        df.index = pd.to_datetime(df.index, errors="coerce")
        df = df.loc[~df.index.isna()].copy()
        df = df.sort_index()

        # Eventuelle Duplikate im Index entfernen
        df = df[~df.index.duplicated(keep="last")]
        if df.empty:
            return None

        # Hinweis: Für "frischen" Cache (<= max_age_hours) reicht die Validierung
        # über das Dateialter (mtime) plus grundlegende Datenbereinigung.
        # Zusätzliche Plausibilitätsprüfungen werden nur bei älteren Cache-Dateien angewandt,
        # um unnötige Live-Abfragen zu vermeiden und dennoch Datenqualität sicherzustellen.
        if max_age_hours <= 24:
            return df

        # Zusätzliche Plausibilitätsprüfung für ältere Cache-Dateien (stale_cache):
        # verhindert die Verwendung stark veralteter oder unvollständiger Daten als Fallback.
        if interval == "1d":
            last = df.index.max()
            if pd.isna(last):
                return None
            if (pd.Timestamp.now() - last) > pd.Timedelta(days=10):
                return None
            if period in {"6mo", "1y"} and len(df) < 60:
                return None
            if period in {"2y", "5y", "max"} and len(df) < 120:
                return None

        return df

    except Exception:
        # Beschädigte oder unbrauchbare Cache-Dateien werden ignoriert
        return None


def _save_to_cache(ticker: str, period: str, interval: str, df: pd.DataFrame) -> None:
    """
    Speichert Kursdaten im Cache.
    Fehler beim Speichern dürfen das Programm nicht abbrechen.
    """
    if df is None or df.empty:
        return
    try:
        df.to_csv(_cache_path(ticker, period, interval))
    except Exception:
        # Cache-Fehler werden bewusst ignoriert
        pass


def load_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
    retries: int = 3,
    backoff_seconds: int = 2,
) -> Tuple[pd.DataFrame, str]:
    """
    Lädt historische Marktdaten für ein Wertpapier oder eine Kryptowährung.

    Features:
    - Lokales Caching zur Reduktion von API-Abfragen
    - Wiederholungsversuche bei Netzwerkfehlern (Retry + Backoff)
    - Fallback auf ältere Cache-Daten bei API-Ausfällen

    Rückgabe:
    - (DataFrame, source) wobei source ∈ {"live", "cache", "stale_cache"}
    """
    # Zuerst versuchen, aktuelle Daten aus dem Cache zu laden
    cached = _load_from_cache(ticker, period, interval)
    if cached is not None and not cached.empty:
        return cached, "cache"

    # yfinance-Ticker mit benutzerdefinierter HTTP-Session initialisieren
    session = _create_session_with_custom_ca()
    stock = yf.Ticker(ticker, session=session)

    last_exc: Optional[Exception] = None

    # Mehrere Abrufversuche mit exponentiellem Backoff
    for attempt in range(retries):
        try:
            df = stock.history(period=period, interval=interval)
            if df is not None and not df.empty:
                df = df.sort_index()

                # Index vereinheitlichen (Live & Cache identisch)
                df.index = pd.to_datetime(df.index, utc=True, errors="coerce").tz_convert(None)
                df = df.loc[~df.index.isna()].copy()
                df = df[~df.index.duplicated(keep="last")]

                _save_to_cache(ticker, period, interval, df)
                return df, "live"

        except Exception as exc:
            last_exc = exc

        time.sleep(backoff_seconds * (2**attempt))

    # Fallback: Nutzung von Cache-Daten bis zu 7 Tage alt
    stale = _load_from_cache(ticker, period, interval, max_age_hours=24 * 7)
    if stale is not None and not stale.empty:
        return stale, "stale_cache"

    # Falls alles fehlschlägt, wird der letzte Fehler weitergereicht
    if last_exc:
        raise last_exc

    raise ValueError(f"Ticker '{ticker}' liefert keine Daten (und kein Cache vorhanden).")