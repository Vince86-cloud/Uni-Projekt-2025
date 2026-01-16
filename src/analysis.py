import pandas as pd
import numpy as np

def _require_close(df: pd.DataFrame) -> None:
    """
    Validiert die Eingabedaten:
    - DataFrame darf nicht leer sein
    - 'Close'-Spalte muss vorhanden sein (Basis für alle Kennzahlen/Indikatoren)
    """
    if df is None or df.empty:
        raise ValueError("DataFrame ist leer")
    if "Close" not in df.columns:
        raise ValueError("Erwarte eine 'Close'-Spalte in den Daten")


def compute_basic_stats(df: pd.DataFrame) -> dict:
    """
    Berechnet grundlegende Kennzahlen auf Basis der Schlusskurse (Close).

    Keys bleiben kompatibel zu bisherigem Code:
    - latest_price: letzter verfügbarer Schlusskurs
    - max_last_year / min_last_year / mean_last_year: Maximum/Minimum/Mittelwert
      (Hinweis: hier über den geladenen Zeitraum des DataFrames, nicht zwingend exakt 1 Jahr)
    
    Zusatzkennzahlen (praktisch für Bericht/Uni):
    - period_return_pct: Rendite über den Zeitraum in Prozent
    - daily_vol_pct: tägliche Volatilität in Prozent (Std der Tagesrenditen)
    """
    _require_close(df)

    close = df["Close"]
    latest_price = close.iloc[-1]

    # Basisstatistiken über den gesamten Zeitraum des DataFrames
    max_val = close.max()
    min_val = close.min()
    mean_val = close.mean()

    # Rendite über den Zeitraum: (Endkurs / Startkurs - 1) * 100
    first_price = close.iloc[0]
    period_return_pct = (latest_price / first_price - 1.0) * 100.0

    # Tagesrenditen (prozentuale Veränderungen); Volatilität = Std der Renditen
    daily_returns = close.pct_change().dropna()
    daily_vol_pct = float(daily_returns.std() * 100.0) if not daily_returns.empty else 0.0

    return {
        "latest_price": float(latest_price),
        "max_last_year": float(max_val),
        "min_last_year": float(min_val),
        "mean_last_year": float(mean_val),
        "period_return_pct": float(period_return_pct),
        "daily_vol_pct": float(daily_vol_pct),
    }


def add_moving_average(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """
    Fügt einen einfachen gleitenden Durchschnitt (SMA) hinzu.
    Ergebnis-Spalte: MA_{window}
    """
    _require_close(df)
    out = df.copy()

    # Rolling-Mean über window Perioden
    out[f"MA_{window}"] = out["Close"].rolling(window=window).mean()
    return out


def add_ema(df: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """
    Fügt einen Exponential Moving Average (EMA) hinzu.
    Ergebnis-Spalte: EMA_{span}
    EMA reagiert stärker auf neue Daten als SMA.
    """
    _require_close(df)
    out = df.copy()

    # Exponentiell gewichteter Mittelwert; adjust=False entspricht typischer Finance-Definition
    out[f"EMA_{span}"] = out["Close"].ewm(span=span, adjust=False).mean()
    return out


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Fügt den RSI (Relative Strength Index) hinzu.
    Implementierung mit Wilder-Glättung:
    - Gewinne/Verluste aus Close-Differenzen
    - geglättete Mittelwerte via EWM (alpha = 1/period)
    Ergebnis-Spalte: RSI_{period} (Wertebereich 0..100)
    """
    _require_close(df)
    out = df.copy()

    # Kursänderung zum Vortag
    delta = out["Close"].diff()

    # Gewinne (positive Änderungen) und Verluste (negative Änderungen) trennen
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    # Wilder smoothing: EWM mit alpha=1/period
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    # RS = avg_gain / avg_loss, RSI = 100 - 100/(1+RS)
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = (100 - (100 / (1 + rs))).astype("float64")

    # Sonderfall: keine Verluste -> RSI = 100
    rsi.loc[avg_loss == 0] = 100.0

    out[f"RSI_{period}"] = rsi
    return out


def add_macd(
    df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9
) -> pd.DataFrame:
    """
    Fügt den MACD-Indikator hinzu:
    - MACD = EMA_fast - EMA_slow
    - MACD_Signal = EMA(MACD, signal)
    - MACD_Hist = MACD - MACD_Signal
    """
    _require_close(df)
    out = df.copy()

    # EMAs des Schlusskurses berechnen
    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()

    # MACD-Linie und Signallinie
    out["MACD"] = ema_fast - ema_slow
    out["MACD_Signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()

    # Histogramm zeigt Abstand zwischen MACD und Signallinie
    out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    return out


def add_indicators(
    df: pd.DataFrame,
    ma_windows=(20,),
    ema_spans=(20, 50, 100),
    rsi_period: int = 14,
    add_macd_flag: bool = True,
) -> pd.DataFrame:
    """
    Fügt mehrere Indikatoren in einem Durchlauf hinzu (nur ein df.copy()).
    Sinnvoll fürs Dashboard, um mehrfaches Kopieren und Rechenaufwand zu vermeiden.

    Enthalten:
    - mehrere SMAs (MA_{w})
    - mehrere EMAs (EMA_{s})
    - RSI (RSI_{rsi_period})
    - optional MACD (MACD, MACD_Signal, MACD_Hist)
    """
    _require_close(df)
    out = df.copy()

    # SMAs
    for w in ma_windows:
        out[f"MA_{w}"] = out["Close"].rolling(window=w).mean()

    # EMAs
    for s in ema_spans:
        out[f"EMA_{s}"] = out["Close"].ewm(span=s, adjust=False).mean()

    # RSI (einmalig berechnen)
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    rsi = rsi.astype(float)
    rsi.loc[avg_loss == 0] = 100.0
    out[f"RSI_{rsi_period}"] = rsi

    # MACD optional hinzufügen
    if add_macd_flag:
        ema_fast = out["Close"].ewm(span=12, adjust=False).mean()
        ema_slow = out["Close"].ewm(span=26, adjust=False).mean()
        out["MACD"] = ema_fast - ema_slow
        out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    return out