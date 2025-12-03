# Hauptprogramm
from data_fetch import load_data
from analysis import compute_basic_stats, add_moving_average
from visualize import plot_history_with_ma


def main() -> None:
    ticker = input("Bitte Ticker eingeben (z.B. AAPL, MSFT, BTC-USD): ").strip().upper()
    if not ticker:
        print("Kein Ticker eingegeben, Programm wird beendet.")
        return

    try:
        df = load_data(ticker)
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return

    df = add_moving_average(df, window=20)
    stats = compute_basic_stats(df)

    print(f"\nBasis-Statistiken fuer {ticker}:")
    print(f"- Letzter Schlusskurs:        {stats['latest_price']:.2f}")
    print(f"- Hoechstkurs (letztes Jahr): {stats['max_last_year']:.2f}")
    print(f"- Tiefstkurs (letztes Jahr):  {stats['min_last_year']:.2f}")
    print(f"- Durchschnitt (letztes Jahr): {stats['mean_last_year']:.2f}")

    plot_history_with_ma(df, ticker)


if __name__ == "__main__":
    main()
