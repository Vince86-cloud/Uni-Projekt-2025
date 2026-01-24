"""
Hauptmenü:

Optionen:
1) Konsolenanalyse (inkl. Matplotlib-Plot)
2) Web-Dashboard (Dash/Plotly)

Wichtig:
- Keine "schweren" Imports (Dash/Matplotlib/yfinance) auf Modulebene
  => Lazy Imports innerhalb der Funktionen verhindern langsame Starts/Importprobleme.
"""

def run_console_mode() -> None:
    # Imports nur wenn wirklich Konsolenmodus gewählt wird
    from src.data_fetch import load_data
    from src.analysis import compute_basic_stats, add_moving_average
    from src.visualize import plot_history_with_ma

    SOURCE_LABELS = {
        "live": "Live-Daten (Yahoo Finance)",
        "cache": "Lokaler Cache",
        "stale_cache": "Älterer Cache (Fallback)",
    }

    ticker = input("Bitte Ticker eingeben (z.B. AAPL, MSFT, BTC-USD): ").strip()
    if not ticker:
        print("Kein Ticker eingegeben, Programm wird beendet.")
        return

    try:
        df, source = load_data(ticker, period="1y", interval="1d")
        if df is None or df.empty:
            print("Keine Daten erhalten (Ticker evtl. ungültig).")
            return
    except Exception as e:
        print(f"Fehler beim Laden der Daten: {e}")
        return

    print(f"\nDatenquelle: {SOURCE_LABELS.get(source, source)}")

    df = add_moving_average(df, window=20)
    stats = compute_basic_stats(df)

    print(f"\nBasis-Statistiken für {ticker}:")
    print(f"- Letzter Schlusskurs:         {stats['latest_price']:.2f}")
    print(f"- Höchstkurs (Zeitraum):       {stats['max_last_year']:.2f}")
    print(f"- Tiefstkurs (Zeitraum):       {stats['min_last_year']:.2f}")
    print(f"- Durchschnitt (Zeitraum):     {stats['mean_last_year']:.2f}")
    print(f"- Rendite (Zeitraum):          {stats['period_return_pct']:.2f}%")
    print(f"- Volatilität (täglich, Std):  {stats['daily_vol_pct']:.2f}%")

    # Offline-Plot (Matplotlib)
    plot_history_with_ma(df, ticker)


def run_dashboard_mode() -> None:
    # Dash-App erst laden, wenn Dashboard wirklich gewählt wird
    from src.dashboard import main as app_main
    app_main()


def main() -> None:
    print("Modus wählen:")
    print("1 = Konsolenanalyse (inkl. Matplotlib-Plot)")
    print("2 = Dashboard (Dash/Plotly)")

    choice = input("Auswahl (1/2): ").strip()

    if choice == "1":
        run_console_mode()
    elif choice == "2":
        run_dashboard_mode()
    else:
        print("Ungültige Auswahl. Programm beendet.")

if __name__ == "__main__":
    main()    