# Datenabruf
import yfinance as yf

def load_data(ticker):
    """
    Lädt historische Kursdaten für den angegebenen Ticker.
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")  # letzte 12 Monate
    return df