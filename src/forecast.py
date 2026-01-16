from __future__ import annotations

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def prepare_series(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """
    Bereitet eine Preis-Zeitreihe für die Prognose mit ARIMA vor.

    Schritte:
    - prüft, ob Daten vorhanden sind
    - extrahiert die gewünschte Spalte (z.B. 'Close')
    - stellt sicher, dass ein DatetimeIndex verwendet wird
    - sortiert die Daten zeitlich
    - entfernt fehlende Werte
    - entfernt Zeitzoneninformationen
    """
    if df is None or df.empty:
        raise ValueError("DataFrame ist leer.")
    if col not in df.columns:
        raise ValueError(f"Erwarte eine '{col}'-Spalte in den Daten.")

    # Zielspalte kopieren (Originaldaten bleiben unverändert)
    s = df[col].copy()

    # Sicherstellen, dass der Index Datumswerte enthält
    if not isinstance(df.index, pd.DatetimeIndex):
        s.index = pd.to_datetime(df.index, errors="coerce")

    # Zeitreihe sortieren und ungültige Werte entfernen
    s = s.sort_index()
    s = s.dropna()

    # Zeitzonen entfernen (statsmodels erwartet tz-naive Indizes)
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    # Mindestanzahl an Datenpunkten für ARIMA sicherstellen
    if len(s) < 30:
        raise ValueError("Zu wenige Datenpunkte für ARIMA (mind. ~30 empfohlen).")

    return s


def arima_forecast(
    df: pd.DataFrame,
    steps: int = 30,
    order: tuple[int, int, int] = (5, 1, 0),
    col: str = "Close",
) -> pd.DataFrame:
    """
    Erstellt eine ARIMA-Prognose für eine gegebene Zeitreihe.

    Rückgabe:
    DataFrame mit:
    - yhat: Prognosemittelwert
    - yhat_lower: untere Grenze des 95%-Konfidenzintervalls
    - yhat_upper: obere Grenze des 95%-Konfidenzintervalls
    """
    # Zeitreihe vorbereiten
    s = prepare_series(df, col=col)

    # ARIMA-Modell fitten
    model = ARIMA(s, order=order)
    fitted = model.fit()

    # Prognose berechnen
    fc = fitted.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    # Zukunftsindex erzeugen (Standard: Geschäftstage)
    # Hinweis: Für Kryptowährungen wäre evtl. "D" sinnvoller
    last_date = s.index[-1]
    future_index = pd.bdate_range(
        start=last_date + pd.Timedelta(days=1), periods=steps
    )

    # Prognosewerte an eigenen Zeitindex anpassen
    mean = pd.Series(mean.values, index=future_index, name="yhat")
    lower = pd.Series(ci.iloc[:, 0].values, index=future_index, name="yhat_lower")
    upper = pd.Series(ci.iloc[:, 1].values, index=future_index, name="yhat_upper")

    # Alles zu einem DataFrame zusammenführen
    out = pd.concat([mean, lower, upper], axis=1)
    return out


def pick_arima_order_quick(
    df: pd.DataFrame, col: str = "Close"
) -> tuple[int, int, int]:
    """
    Einfache Heuristik zur Wahl der ARIMA-Modellordnung (p,d,q).

    - kurze Zeitreihen -> einfaches Modell
    - längere Zeitreihen -> etwas komplexeres Modell

    Ziel: stabile Prognose ohne aufwändige Modellselektion
    """
    s = prepare_series(df, col=col)

    # Bei kurzen Zeitreihen Komplexität reduzieren
    if len(s) < 120:
        return (2, 1, 0)

    return (5, 1, 0)