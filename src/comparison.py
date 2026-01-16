from src.data_fetch import load_data
import pandas as pd
import numpy as np
import datetime

# Die Klasse repräsentiert ein einzelnes Finanz-Asset (Aktie/Krypto etc.)
# und kapselt relevante Kennzahlen und Auswertungen zu genau einem Asset.
# Sie dient als "Container" für den Vergleich mehrerer Assets.
class Asset:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df

        # Index sicher als DatetimeIndex (UTC) setzen, damit Zeitvergleiche robust sind
        self.df.index = pd.to_datetime(self.df.index, utc=True)

    # Validiert einen absoluten Datumsbereich (Start- und Enddatum).
    # Prüft Datumsformat, Zeitzonenkonsistenz, Reihenfolge und ob Daten im Zeitraum vorhanden sind.
    def _validate_date_range(self, start_date, end_date):
        # Datumsstrings/Objekte in Timestamps konvertieren
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except Exception as e:
            raise ValueError("Geben Sie ein gültiges Datum ein") from e

        if pd.isna(start_date) or pd.isna(end_date):
            raise ValueError("Start- und Enddatum dürfen nicht leer sein")

        # Sicherstellen, dass Start- und Enddatum dieselbe Zeitzone wie der DataFrame-Index haben
        tz = self.df.index.tz
        if tz is not None:
            # Wenn das Datum tz-naiv ist, lokalisieren; sonst in die Index-Zeitzone konvertieren
            if start_date.tzinfo is None:
                start_date = start_date.tz_localize(tz)
            else:
                start_date = start_date.tz_convert(tz)

            if end_date.tzinfo is None:
                end_date = end_date.tz_localize(tz)
            else:
                end_date = end_date.tz_convert(tz)

        # Start muss vor End liegen
        if start_date >= end_date:
            raise ValueError("Startdatum muss vor Enddatum liegen")

        # Zeitraum aus den Daten ausschneiden
        period = self.df.loc[start_date:end_date]
        if period.empty:
            raise ValueError("Keine Daten im angegebenen Zeitraum")

        # Für Renditen etc. braucht man mindestens zwei Datenpunkte
        if len(period) < 2:
            raise ValueError("Zeitraum enthält weniger als zwei Handelstage")

        return period

    # Validiert einen relativen Zeitraum in Tagen.
    # Prüft Typ, Wertebereich und ob der Zeitraum durch die vorhandenen Daten abgedeckt ist.
    def _validate_days(self, days):
        if not isinstance(days, (int, float)):
            raise ValueError("Tage müssen eine Zahl sein")

        if days <= 0:
            raise ValueError("Tage müssen größer als 0 sein")

        # Größtes verfügbares Datum in den Daten (nicht zwingend "heute")
        today = self.df.index.max()
        first_date = self.df.index.min()

        # Wenn days größer als verfügbar: auf maximal möglichen Zeitraum begrenzen
        max_days = (today - first_date).days
        days = int(days)
        days = min(days, max_days)

        # Startdatum relativ zum letzten verfügbaren Datum bestimmen
        start_date = today - pd.Timedelta(days=days)
        period = self.df.loc[start_date:today]

        if len(period) < 2:
            raise ValueError("Zeitraum enthält weniger als zwei Handelstage")

        return period

    # Liefert den relevanten Zeitraum (DataFrame-Ausschnitt) für Kennzahlberechnungen.
    # Entweder:
    # - absolut: start_date UND end_date
    # - relativ: days
    # Es wird sichergestellt, dass genau eine Variante verwendet wird.
    def _get_period(self, start_date=None, end_date=None, days=None):
        if days is not None:
            return self._validate_days(days)
        elif start_date is not None and end_date is not None:
            return self._validate_date_range(start_date, end_date)
        else:
            raise ValueError("Entweder start_date & end_date oder days angeben")

    # Berechnet die absolute Preisentwicklung (Endkurs - Startkurs) im Zeitraum.
    def price_development(self, start_date=None, end_date=None, days=None):
        period = self._get_period(start_date=start_date, end_date=end_date, days=days)
        first_close = period["Close"].iloc[0]
        last_close = period["Close"].iloc[-1]
        price_development = last_close - first_close
        return price_development

    # Berechnet die prozentuale Rendite ((End - Start) / Start * 100) im Zeitraum.
    def return_percentage(self, start_date=None, end_date=None, days=None):
        period = self._get_period(start_date=start_date, end_date=end_date, days=days)
        first_close = period["Close"].iloc[0]
        last_close = period["Close"].iloc[-1]
        return_percentage = ((last_close - first_close) / first_close) * 100
        return return_percentage

    # Normalisiert die Close-Reihe auf 100 am Start (vergleichbar über Assets mit unterschiedlichen Preisen).
    def normalized_price_series(self, start_date=None, end_date=None, days=None):
        period = self._get_period(start_date=start_date, end_date=end_date, days=days)
        close = period["Close"]
        price_series = (close / close.iloc[0]) * 100
        return price_series

    # Gesamtperformance als Endwert der normalisierten Reihe (Start=100 -> Endwert z.B. 135).
    def normalized_performance(self, start_date=None, end_date=None, days=None):
        normalized_series = self.normalized_price_series(
            start_date=start_date, end_date=end_date, days=days
        )
        end_value = normalized_series.iloc[-1]
        return end_value

    # Annualisierte Volatilität basierend auf täglichen Renditen.
    # Annahme: 252 Handelstage pro Jahr (klassische Finanz-Annahme).
    def volatility(self, start_date=None, end_date=None, days=None):
        period = self._get_period(start_date=start_date, end_date=end_date, days=days)
        close = period["Close"]
        daily_returns = close.pct_change()
        volatility = daily_returns.std(ddof=1) * np.sqrt(252)
        return volatility

    # Maximaler Drawdown: größter relativer Verlust vom bisherigen Hoch zum folgenden Tief.
    def drawdown(self, start_date=None, end_date=None, days=None):
        period = self._get_period(start_date=start_date, end_date=end_date, days=days)
        close = period["Close"]

        # Tagesrenditen -> kumulative Renditen -> kumulatives Maximum -> Drawdown
        daily_returns = close.pct_change()
        daily_returns.fillna(0.0, inplace=True)
        cumulative_returns = (1 + daily_returns).cumprod()
        cumulative_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        return max_drawdown

    # Bester Tag: größter Tagesgewinn (maximale Tagesrendite) im Zeitraum.
    def best_day(self, start_date=None, end_date=None, days=None):
        period = self._get_period(start_date=start_date, end_date=end_date, days=days)
        close = period["Close"]
        daily_returns = close.pct_change()
        best_return = daily_returns.max()
        best_date = daily_returns.idxmax()
        return {"return": best_return, "date": best_date}

    # Schlechtester Tag: größter Tagesverlust (minimale Tagesrendite) im Zeitraum.
    def worst_day(self, start_date=None, end_date=None, days=None):
        period = self._get_period(start_date=start_date, end_date=end_date, days=days)
        close = period["Close"]
        daily_returns = close.pct_change()
        worst_return = daily_returns.min()
        worst_date = daily_returns.idxmin()
        return {"return": worst_return, "date": worst_date}

    # Bündelt alle Kennzahlen eines Assets in einem Dictionary.
    # Enthält sowohl skalare Werte als auch die normierte Zeitreihe.
    def summary_metrics(self, start_date=None, end_date=None, days=None):
        return {
            "price_development": self.price_development(
                start_date=start_date, end_date=end_date, days=days
            ),
            "return_percentage": self.return_percentage(
                start_date=start_date, end_date=end_date, days=days
            ),
            "normalized_price_series": self.normalized_price_series(
                start_date=start_date, end_date=end_date, days=days
            ),
            "normalized_performance": self.normalized_performance(
                start_date=start_date, end_date=end_date, days=days
            ),
            "volatility": self.volatility(start_date=start_date, end_date=end_date, days=days),
            "drawdown": self.drawdown(start_date=start_date, end_date=end_date, days=days),
            "best_day": self.best_day(start_date=start_date, end_date=end_date, days=days),
            "worst_day": self.worst_day(start_date=start_date, end_date=end_date, days=days),
        }


# Die Klasse repräsentiert einen Vergleich mehrerer Finanz-Assets.
# Sie vergleicht zentrale Kennzahlen (z.B. Rendite, Volatilität, Drawdown)
# der Assets über identische Zeiträume und stellt die Ergebnisse strukturiert dar.
class Comparator:
    def __init__(self, assets):
        self.assets = assets

    # Interpretiert die Differenz einer Kennzahl zwischen zwei Assets.
    # Für Risiko-Kennzahlen (Volatilität, Drawdown) gilt: kleiner = besser.
    # Für Performance-Kennzahlen gilt: größer = besser.
    def _interpret_metric(self, metric, diff):
        if metric in ["volatility", "drawdown"]:
            return "besser" if diff < 0 else "schlechter"
        else:
            return "besser" if diff > 0 else "schlechter"

    # Bestimmt bei >2 Assets das "beste" Asset pro Kennzahl.
    # Risiko-Kennzahlen: Minimum ist besser; Performance-Kennzahlen: Maximum ist besser.
    def _best_asset_per_metric(self, row):
        metric = row.name

        if metric in ["volatility", "drawdown"]:
            return row.idxmin().replace(" (Abw. %)", "")
        else:
            return row.idxmax().replace(" (Abw. %)", "")

    # Ruft summary_metrics() je Asset auf und sammelt alle Kennzahlen für den Vergleich.
    def collect_metrics(self, start_date=None, end_date=None, days=None):
        results = {}
        for asset in self.assets:
            results[asset.ticker] = asset.summary_metrics(
                start_date=start_date, end_date=end_date, days=days
            )
        return results

    # Vergleicht Assets anhand skalare Kennzahlen und erzeugt eine Vergleichstabelle.
    def compare_metrics(self, metrics):
        if len(metrics) < 2:
            raise ValueError("Für einen Assetvergleich müssen mindestens 2 Assets angegeben werden")

        dic = {}

        # Nur skalare Werte (int/float) übernehmen, Zeitreihen/Dictionaries werden ausgelassen
        for ticker, metric_dict in metrics.items():
            scalar_assets = {}
            for metrics_name, metrics_value in metric_dict.items():
                if isinstance(metrics_value, (int, float, np.number)):
                    scalar_assets[metrics_name] = metrics_value
            dic[ticker] = scalar_assets

        # Tabelle: Zeilen = Kennzahlen, Spalten = Assets
        raw_scalar_df = pd.DataFrame.from_dict(data=dic, orient="index")
        compare_df = raw_scalar_df.T

        # Fall 1: genau zwei Assets -> Differenz + relative Abweichung + Textbewertung
        if len(metrics) == 2:
            asset_a, asset_b = compare_df.columns[0], compare_df.columns[1]

            # Differenz = B - A
            compare_df["Differenz"] = compare_df[asset_b] - compare_df[asset_a]

            # Relative Abweichung bezogen auf Asset A (Schutz gegen Division durch 0)
            denom = compare_df[asset_a].abs().replace(0, np.nan)
            compare_df["Relative Abweichung (%)"] = (compare_df["Differenz"] / denom) * 100

            # Qualitative Bewertung je Kennzahl
            compare_df["Bewertung"] = [
                f"{asset_b} {self._interpret_metric(metric, diff)} als {asset_a}"
                for metric, diff in zip(compare_df.index, compare_df["Differenz"])
            ]
            return compare_df

        # Fall 2: mehr als zwei Assets -> Vergleich relativ zum Mittelwert + bestes Asset
        else:
            mean_series = compare_df.mean(axis=1)

            # relative Abweichung vom Mittelwert (in %)
            denom = mean_series.abs().replace(0, np.nan)
            relative_df = (compare_df.sub(mean_series, axis=0).div(denom, axis=0) * 100)

            # Spaltennamen kennzeichnen, dass es Abweichungen in Prozent sind
            relative_df.columns = [f"{c} (Abw. %)" for c in relative_df.columns]

            # "Bewertung": bestes Asset pro Kennzahl
            relative_df["Bewertung"] = relative_df.apply(self._best_asset_per_metric, axis=1)
            return relative_df
            
            



        
            

        
    







            
            

        

        



        
        

        
        
        
        

         
         

       

        
        

        




        
        
        
        

        

        

        
        
        

        

        


        

        

        

        
        
        
        
        




        

        






        
        
        
        


