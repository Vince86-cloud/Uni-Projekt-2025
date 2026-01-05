from __future__ import annotations

import pandas as pd
from statsmodels.tsa.arima.model import ARIMA


def prepare_series(df: pd.DataFrame, col: str = "Close") -> pd.Series:
    """
    Extract and clean a price series for forecasting.
    - ensures DatetimeIndex
    - sorts index
    - drops NaNs
    """
    if df is None or df.empty:
        raise ValueError("DataFrame ist leer.")
    if col not in df.columns:
        raise ValueError(f"Erwarte eine '{col}'-Spalte in den Daten.")

    s = df[col].copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        
        s.index = pd.to_datetime(df.index, errors="coerce")

    s = s.sort_index()
    s = s.dropna()

   
    if hasattr(s.index, "tz") and s.index.tz is not None:
        s.index = s.index.tz_localize(None)

    if len(s) < 30:
        raise ValueError("Zu wenige Datenpunkte für ARIMA (mind. ~30 empfohlen).")

    return s


def arima_forecast(
    df: pd.DataFrame,
    steps: int = 30,
    order: tuple[int, int, int] = (5, 1, 0),
    col: str = "Close",
) -> pd.DataFrame:

    s = prepare_series(df, col=col)

    # Fit model
    model = ARIMA(s, order=order)
    fitted = model.fit()

    # Forecast
    fc = fitted.get_forecast(steps=steps)
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)

    # Build future index (business days). For crypto, you might prefer "D".
    last_date = s.index[-1]
    future_index = pd.bdate_range(start=last_date + pd.Timedelta(days=1), periods=steps)

    # Align to our future index (statsmodels uses integer index sometimes)
    mean = pd.Series(mean.values, index=future_index, name="yhat")
    lower = pd.Series(ci.iloc[:, 0].values, index=future_index, name="yhat_lower")
    upper = pd.Series(ci.iloc[:, 1].values, index=future_index, name="yhat_upper")

    out = pd.concat([mean, lower, upper], axis=1)
    return out


def pick_arima_order_quick(df: pd.DataFrame, col: str = "Close") -> tuple[int, int, int]:
   
    s = prepare_series(df, col=col)

    # If the series is short, keep it simpler
    if len(s) < 120:
        return (2, 1, 0)
    return (5, 1, 0)