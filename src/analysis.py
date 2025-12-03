import pandas as pd


def compute_basic_stats(df: pd.DataFrame) -> dict:
    """Compute basic statistics for closing prices (last row, min, max, mean)."""
    if "Close" not in df.columns:
        raise ValueError("Erwarte eine 'Close'-Spalte in den Daten")

    latest_price = df["Close"].iloc[-1]
    max_last_year = df["Close"].max()
    min_last_year = df["Close"].min()
    mean_last_year = df["Close"].mean()

    return {
        "latest_price": float(latest_price),
        "max_last_year": float(max_last_year),
        "min_last_year": float(min_last_year),
        "mean_last_year": float(mean_last_year),
    }


def add_moving_average(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Add a simple moving average of closing prices."""
    if "Close" not in df.columns:
        raise ValueError("Erwarte eine 'Close'-Spalte in den Daten")

    df = df.copy()
    df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()
    return df
