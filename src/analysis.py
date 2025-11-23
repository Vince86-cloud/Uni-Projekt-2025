# Datenanalyse
def calculate_statistics(df):
    """
    Berechnet Höchst-, Tiefst- und Durchschnittspreis.
    """
    high = df['High'].max()
    low = df['Low'].min()
    avg = df['Close'].mean()
    return {"high": high, "low": low, "average": avg}