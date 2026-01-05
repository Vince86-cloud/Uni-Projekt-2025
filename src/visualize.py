import matplotlib.pyplot as plt
import pandas as pd


def _require_close(df: pd.DataFrame) -> None:
    if df is None or df.empty:
        raise ValueError("DataFrame ist leer")
    if "Close" not in df.columns:
        raise ValueError("Erwarte eine 'Close'-Spalte in den Daten")


def _normalize_index(df: pd.DataFrame) -> pd.DataFrame:
    # falls tz-aware index: Matplotlib kann sonst nerven
    if hasattr(df.index, "tz") and df.index.tz is not None:
        out = df.copy()
        out.index = out.index.tz_localize(None)
        return out
    return df


def plot_data(df: pd.DataFrame) -> None:
    """Einfacher Close-Plot."""
    _require_close(df)
    df = _normalize_index(df)

    df["Close"].plot(title="Kursverlauf")
    plt.xlabel("Datum")
    plt.ylabel("Schlusskurs")
    plt.tight_layout()
    plt.show()


def plot_history_with_ma(df: pd.DataFrame, ticker: str, window: int = 20) -> None:
    """
    Rückwärtskompatibel: Plottet Close + alle MA_* Spalten.
    (window bleibt für alte API erhalten)
    """
    _require_close(df)
    df = _normalize_index(df)

    def _sort_suffix(cols, prefix):
        def key(c):
            s = c.replace(prefix, "")
            return int(s) if s.isdigit() else 10**9
        return sorted(cols, key=key)

    ma_cols = _sort_suffix([col for col in df.columns if col.startswith("MA_")], "MA_")

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["Close"], label="Close", linewidth=1.5)

    for col in ma_cols:
        plt.plot(df.index, df[col], label=col, linewidth=1.2)

    plt.title(f"{ticker} Kursverlauf mit gleitendem Durchschnitt")
    plt.xlabel("Datum")
    plt.ylabel("Schlusskurs")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_history_with_indicators(
    df: pd.DataFrame,
    ticker: str,
    show_ma: bool = True,
    show_ema: bool = True,
    show_rsi: bool = True,
    show_macd: bool = True,
) -> None:
    """
    Panel 1: Close + MA_* + EMA_*
    Panel 2: RSI_*
    Panel 3: MACD
    """
    _require_close(df)
    df = _normalize_index(df)

    def _sort_suffix(cols, prefix):
        def key(c):
            s = c.replace(prefix, "")
            return int(s) if s.isdigit() else 10**9
        return sorted(cols, key=key)

    ma_cols = _sort_suffix([c for c in df.columns if c.startswith("MA_")], "MA_") if show_ma else []
    ema_cols = _sort_suffix([c for c in df.columns if c.startswith("EMA_")], "EMA_") if show_ema else []
    rsi_cols = _sort_suffix([c for c in df.columns if c.startswith("RSI_")], "RSI_") if show_rsi else []

    has_rsi = len(rsi_cols) > 0
    has_macd = show_macd and all(c in df.columns for c in ["MACD", "MACD_Signal", "MACD_Hist"])

    n_panels = 1 + (1 if has_rsi else 0) + (1 if has_macd else 0)

    if n_panels == 1:
        fig, ax_price = plt.subplots(1, 1, figsize=(11, 5))
        axes = [ax_price]
    elif n_panels == 2:
        fig, (ax_price, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11, 7), gridspec_kw={"height_ratios": [3, 1]})
        axes = [ax_price, ax2]
    else:
        fig, (ax_price, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(11, 9), gridspec_kw={"height_ratios": [3, 1, 1]})
        axes = [ax_price, ax2, ax3]

    # Panel 1
    ax_price = axes[0]
    ax_price.plot(df.index, df["Close"], label="Close", linewidth=1.6)

    for col in ma_cols:
        ax_price.plot(df.index, df[col], label=col, linewidth=1.2)
    for col in ema_cols:
        ax_price.plot(df.index, df[col], label=col, linewidth=1.2)

    ax_price.set_title(f"{ticker} – Kurs & Indikatoren")
    ax_price.set_ylabel("Preis")
    ax_price.grid(True, alpha=0.3)
    ax_price.legend()

    panel_idx = 1

    # RSI
    if has_rsi:
        ax_rsi = axes[panel_idx]
        rsi_col = "RSI_14" if "RSI_14" in rsi_cols else rsi_cols[0]
        ax_rsi.plot(df.index, df[rsi_col], label=rsi_col, linewidth=1.2)
        ax_rsi.axhline(70, linestyle="--")
        ax_rsi.axhline(30, linestyle="--")
        ax_rsi.set_ylim(0, 100)
        ax_rsi.set_ylabel("RSI")
        ax_rsi.grid(True, alpha=0.3)
        ax_rsi.legend(loc="upper left")
        panel_idx += 1

    # MACD
    if has_macd:
        ax_macd = axes[panel_idx]
        ax_macd.bar(df.index, df["MACD_Hist"], label="MACD Hist", alpha=0.4)
        ax_macd.plot(df.index, df["MACD"], label="MACD", linewidth=1.2)
        ax_macd.plot(df.index, df["MACD_Signal"], label="Signal", linewidth=1.2)
        ax_macd.axhline(0, linestyle="--", linewidth=1)
        ax_macd.set_ylabel("MACD")
        ax_macd.grid(True, alpha=0.3)
        ax_macd.legend(loc="upper left")

    plt.tight_layout()
    plt.show()