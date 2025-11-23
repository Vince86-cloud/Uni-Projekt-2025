# Hauptprogramm
from data_fetch import load_data
from analysis import calculate_statistics
from visualize import plot_data

def main():
    ticker = input("Bitte Ticker eingeben: ")
    df = load_data(ticker)
    stats = calculate_statistics(df)
    
    print(f"Höchstkurs letztes Jahr: {stats['high']}")
    print(f"Tiefstkurs letztes Jahr: {stats['low']}")
    print(f"Durchschnittskurs: {stats['average']:.2f}")
    
    plot_data(df)

if __name__ == "__main__":
    main()