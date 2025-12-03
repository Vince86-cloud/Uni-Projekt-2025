import matplotlib.pyplot as plt
import pandas as pd

def plot_data(df: pd.DataFrame) -> None:
    if "Close" not in df.columns:
        raise ValueError("Erwarte eine 'Close'-Spalte in den Daten")
    df["Close"].plot(title="Kursverlauf")
    plt.xlabel("Datum")
    plt.ylabel("Schlusskurs")
    plt.tight_layout()
    plt.show()

def plot_history_with_ma(df: pd.DataFrame, ticker: str, window: int = 20) -> None:
    if "Close" not in df.columns:
        raise ValueError("Erwarte eine 'Close'-Spalte in den Daten")

    ma_cols = [col for col in df.columns if col.startswith("MA_")]
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
