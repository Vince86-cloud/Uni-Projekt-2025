#Visualisierung
import matplotlib.pyplot as plt

def plot_data(df):
    """
    Erstellt ein Liniendiagramm des Schlusskurses.
    """
    df['Close'].plot(title="Kursverlauf")
    plt.xlabel("Datum")
    plt.ylabel("Schlusskurs")
    plt.show()