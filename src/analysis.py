import pandas as pd

def _require_close(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("DataFrame ist leer")
    if "Close" not in df.columns:
        raise ValueError("Erwarte eine 'Close'-Spalte in den Daten")


def compute_basic_stats(df: pd.DataFrame) -> dict:
    """
    Compute basic statistics for closing prices over the provided DataFrame.

    Keys bleiben kompatibel zu deinem bisherigen Code:
    - latest_price
    - max_last_year
    - min_last_year
    - mean_last_year
    """
    _require_close(df)

    close = df["Close"]
    latest_price = close.iloc[-1]

    # "last_year" ist hier: Zeitraum des DataFrames (z.B. 1y, 3mo, etc.)
    max_val = close.max()
    min_val = close.min()
    mean_val = close.mean()

    # Bonus-Kennzahlen (praktisch für Bericht/Uni):
    first_price = close.iloc[0]
    period_return_pct = (latest_price / first_price - 1.0) * 100.0

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
    """Add a simple moving average (SMA) of closing prices."""
    _require_close(df)
    out = df.copy()
    out[f"MA_{window}"] = out["Close"].rolling(window=window).mean()
    return out


def add_ema(df: pd.DataFrame, span: int = 20) -> pd.DataFrame:
    """Add Exponential Moving Average (EMA) for closing prices."""
    _require_close(df)
    out = df.copy()
    out[f"EMA_{span}"] = out["Close"].ewm(span=span, adjust=False).mean()
    return out


def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """Add RSI (Relative Strength Index) using Wilder's smoothing."""
    _require_close(df)
    out = df.copy()

    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = (100 - (100 / (1 + rs))).astype("float64")
    rsi.loc[avg_loss == 0] = 100.0

    out[f"RSI_{period}"] = rsi
    return out


def add_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Add MACD indicator (MACD, MACD_Signal, MACD_Hist)."""
    _require_close(df)
    out = df.copy()

    ema_fast = out["Close"].ewm(span=fast, adjust=False).mean()
    ema_slow = out["Close"].ewm(span=slow, adjust=False).mean()

    out["MACD"] = ema_fast - ema_slow
    out["MACD_Signal"] = out["MACD"].ewm(span=signal, adjust=False).mean()
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
    Add typical indicators in one go (only one df.copy()).
    Useful for dashboard to avoid repeated copying.
    """
    _require_close(df)
    out = df.copy()

    for w in ma_windows:
        out[f"MA_{w}"] = out["Close"].rolling(window=w).mean()

    for s in ema_spans:
        out[f"EMA_{s}"] = out["Close"].ewm(span=s, adjust=False).mean()

    # RSI
    delta = out["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / rsi_period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / rsi_period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = (100 - (100 / (1 + rs))).astype("float64")
    rsi.loc[avg_loss == 0] = 100.0
    out[f"RSI_{rsi_period}"] = rsi

    if add_macd_flag:
        ema_fast = out["Close"].ewm(span=12, adjust=False).mean()
        ema_slow = out["Close"].ewm(span=26, adjust=False).mean()
        out["MACD"] = ema_fast - ema_slow
        out["MACD_Signal"] = out["MACD"].ewm(span=9, adjust=False).mean()
        out["MACD_Hist"] = out["MACD"] - out["MACD_Signal"]

    return out