from data_fetch import load_data
import pandas as pd
import numpy as np
import datetime 


# Die Klasse repräsentiert ein einzelnes Finanz-Asset
# und kapselt relevante Kennzahlen und Auswertungen zu genau einem Asset zusammen. 
# Sie dient als Container für den Vergleich mehrerer Assets.
class Asset:
    def __init__(self, ticker, df):
        self.ticker = ticker
        self.df = df

    # Validiert einen absoluten Datumsbereich mit einem Start- und Enddatum.
    # Prüft Datumsformat, Reihenfolge und ob die Daten im angegebenen Zeitraum vorhanden sind.
    def _validate_date_range(self, start_date, end_date):
        try:
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
        except:
            raise ValueError("Geben Sie ein gültiges Datum ein")   

        if pd.isna(start_date) or pd.isna(end_date):
            raise ValueError("Start- und Enddatum dürfen nicht leer sein")
        
        if start_date >= end_date:
            raise ValueError("Startdatum muss vor Enddatum liegen")

        period = self.df.loc[start_date:end_date]
            
        if period.empty:
            raise ValueError("Keine Daten im angegebenen Zeitraum")

        if len(period) < 2:
            raise ValueError("Zeitraum enthält weniger als zwei Handelstage")

        return period


    # Validiert einen relativen Zeitraum in Tagen.
    # Prüft Typ, Wertebereich und ob der Zeitraum durch die vorhandenen Daten abgedeckt ist
    def _validate_days(self, days):
        if not isinstance(days, (int,float)):
            raise ValueError("Tage müssen eine Zahl sein")
            
        if days <= 0:
            raise ValueError("Tage müssen größer als 0 sein")

        today = self.df.index.max()
        first_date = self.df.index.min()
        max_days = (today - first_date).days

        if days > max_days:
            raise ValueError("Keine Daten im angegebenen Zeitraum")

        start_date = today - pd.Timedelta(days=days)
        period = self.df.loc[start_date:today]

        if len(period) < 2:
            raise ValueError("Zeitraum enthält weniger als zwei Handelstage")
            
        return period
        
    # Berechnet die Preisentwicklung eines Assets über einen bestimmten Zeitraum
    # auf Basis der Close-Preise.
    def price_development(self,start_date, end_date):
        period = self._validate_date_range(start_date, end_date)
        first_close = period['Close'].iloc[0]
        last_close  = period['Close'].iloc[-1]
        price_development = last_close - first_close
        return price_development

    # Berechnet die Preisentwicklung eines Assets über einen relativen Zeitraum
    # (z.B. letzte 30/90/365 Tage) ausgehend vom aktuellen Datum
    # auf Basis der Close-Preise.
    def price_development_period(self, days):
        period = self._validate_days(days)
        first_close = period['Close'].iloc[0]
        last_close  = period['Close'].iloc[-1]
        price_development_period = last_close - first_close
        return price_development_period

    # Berechnet die prozentuale Rendite eines Assets über einen bestimmten Zeitraum
    # auf Basis der Close-Preise.
    def return_percentage(self, start_date, end_date):
        period = self._validate_date_range(start_date, end_date)
        first_close = period['Close'].iloc[0]
        last_close  = period['Close'].iloc[-1]
        return_percentage = ((last_close - first_close) / first_close) * 100
        return return_percentage

    # Berechnet die prozentuale Preisentwicklung eines Assets über einen relativen Zeitraum
    # (z.B. letzte 30/90/365 Tage) ausgehend vom aktuellen Datum
    # auf Basis der Close-Preise.
    def return_percentage_period(self,days):
        period = self._validate_days(days)
        first_close = period['Close'].iloc[0]
        last_close  = period['Close'].iloc[-1]
        return_percentage_period = ((last_close - first_close) / first_close) * 100
        return return_percentage_period

    # Berechnet den Kursverlauf eines Assets auf 100 normalisiert über einen bestimmten Zeitraum
    # auf Basis der Close-Preise.
    def normalized_price_series(self,start_date, end_date):
        period = self._validate_date_range(start_date, end_date)
        close = period['Close']
        price_series = (close / close.iloc[0]) * 100
        return price_series

    # Berechnet den Kursverlauf eines Assets auf 100 normalisiert über einen relativen Zeitraum
    # (z.B. letzte 30/90/365 Tage) ausgehend vom aktuellen Datum
    # auf Basis der Close-Preise.
    def normalized_price_series_period(self, days):
        period = self._validate_days(days)
        close = period['Close']
        price_series = (close / close.iloc[0]) * 100
        return price_series

    # Berechnet die annualisierte Volatilität eines Assets über einen bestimmten Zeitraum
    # auf Basis der Close-Preise.
    def volatility(self, start_date, end_date):
        period = self._validate_date_range(start_date, end_date)
        close = period['Close']
        daily_returns = close.pct_change()
        volatility = daily_returns.std(ddof=1) * np.sqrt(252)
        return volatility

    # Berechnet die annualisierte Volatilität eines Assets über einen relativen Zeitraum
    # (z.B. letzte 30/90/365 Tage) ausgehend vom aktuellen Datum
    # auf Basis der Close-Preise.
    def volatility_period(self, days):
        period = self._validate_days(days)
        close = period['Close']
        daily_returns = close.pct_change()
        volatility = daily_returns.std(ddof=1) * np.sqrt(252)
        return volatility

    # Berechnet den Verlust eines Assets vom letzten Höchststand bis zu einem späteren Tiefpunkt (maximalen Drawdown)
    # über einen bestimmten Zeitraum auf Basis der Close-Preise
    def drawdown(self, start_date, end_date):
        period = self._validate_date_range(start_date, end_date)
        close = period['Close']
        daily_returns = close.pct_change()
        daily_returns.fillna(0.0, inplace=True)
        cumulative_returns = (1 + daily_returns).cumprod()
        cumulative_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        return max_drawdown

    # Berechnet den Verlust eines Assets vom letzten Höchststand bis zu einem späteren Tiefpunkt (maximalen Drawdown)
    # über einen relativen Zeitraum (z.B. letzte 30/90/365 Tage) ausgehend vom aktuellen Datum
    # auf Basis der Close-Preise.
    def drawdown_series(self, days):
        period = self._validate_days(days)
        close = period['Close']
        daily_returns = close.pct_change()
        daily_returns.fillna(0.0, inplace=True)
        cumulative_returns = (1 + daily_returns).cumprod()
        cumulative_max = cumulative_returns.cummax()
        drawdown = (cumulative_returns - cumulative_max) / cumulative_max
        max_drawdown = drawdown.min()
        return max_drawdown


    # Berechnet den besten Tagesgewinn eines Assets über einen bestimmten Zeitraum
    # auf Basis der täglichen Renditen aus den Close-Preisen.
    def best_day(self,start_date, end_date):
        period = self._validate_date_range(start_date, end_date)
        close = period['Close']
        daily_returns = close.pct_change()
        best_return = daily_returns.max()
        best_date = daily_returns.idxmax()
        return {"return": best_return,
                "date": best_date
               }

    # Berechnet den besten Tagesgewinn eines Assets über einen relativen Zeitraum 
    # (z.B. letzte 30/90/365 Tage) ausgehend vom aktuellen Datum auf Basis
    # der täglichen Renditen aus den Close-Preise.
    def best_day_period(self, days):
        period = self._validate_days(days)
        close = period['Close']
        daily_returns = close.pct_change()
        best_return = daily_returns.max()
        best_date = daily_returns.idxmax()
        return {"return": best_return,
                "date": best_date
               }

    # Berechnet den schlechtesten Tagesverlust eines Assets über einen bestimmten Zeitraum
    # auf Basis der täglichen Renditen aus den Close-Preisen.
    def worst_day(self,start_date, end_date):
        period = self._validate_date_range(start_date, end_date)
        close = period['Close']
        daily_returns = close.pct_change()
        worst_return = daily_returns.min()
        worst_date = daily_returns.idxmin()
        return {"return": worst_return,
                "date": worst_date
               }

    # Berechnet den schlechtesten Tagesverlust eines Assets über einen relativen Zeitraum 
    # (z.B. letzte 30/90/365 Tage) ausgehend vom aktuellen Datum auf Basis
    # der täglichen Renditen aus den Close-Preise.
    def worst_day_period(self, days):
        period = self._validate_days(days)
        close = period['Close']
        daily_returns = close.pct_change()
        worst_return = daily_returns.min()
        worst_date = daily_returns.idxmin()
        return {"return": worst_return,
                "date": worst_date
               }

    # Fasst alle Kennzahlen eines einzelnen Assets, für einen absoluten (Start-/Enddatum)
    # oder relativen Zeitraum (Tage) gebündelt zusammen und gibt die Ergebnisse 
    # als Dictionary zurück
    def summary_metrics(self, start_date=None, end_date=None, days=None):
        if (start_date is not None and end_date is not None) and days is None:
            price_development = self.price_development(start_date, end_date)
            return_percentage = self.return_percentage(start_date, end_date)
            normalized_price_series = self.normalized_price_series(start_date, end_date)
            volatility = self.volatility(start_date, end_date)
            drawdown = self.drawdown(start_date, end_date)
            best_day = self.best_day(start_date, end_date)
            worst_day = self.worst_day(start_date, end_date)

            return {"price_development": price_development,
                    "return_percentage": return_percentage,
                    "normalized_price_series": normalized_price_series,
                    "volatility": volatility,
                    "drawdown": drawdown,
                    "best_day":best_day,
                    "worst_day": worst_day
                   }
            
        elif days is not None and (start_date is None and end_date is None):
            price_development_period = self.price_development_period(days)
            return_percentage_period = self.return_percentage_period(days)
            normalized_price_series_period = self.normalized_price_series_period(days)
            volatility_period = self.volatility_period(days)
            drawdown_series = self.drawdown_series(days)
            best_day_period =  self.best_day_period(days)
            worst_day_period = self.worst_day_period(days)

            return {"price_development_period": price_development_period,
                    "return_percentage_period": return_percentage_period,
                    "normalized_price_series_period": normalized_price_series_period,
                    "volatility_period": volatility_period,
                    "drawdown_series": drawdown_series,
                    "best_day_period":best_day_period,
                    "worst_day_period": worst_day_period
                   }
        else:
            raise ValueError("Entweder einen start_date & end_date oder days angeben")


# Die Klasse repräsentiert einen Vergleich mehrerer Finanz-Assets.
# Sie vergleicht zentrale Kennzahlen (z.B. Rendite, Volatilität, Drawdown)
# der Assets über identische Zeiträume und stellt die Ergebnisse strukturiert dar,
# um Unterschiede zwischen den Assets transparent zu machen. 






            
            

        

        



        
        

        
        
        
        

         
         

       

        
        

        




        
        
        
        

        

        

        
        
        

        

        


        

        

        

        
        
        
        
        




        

        






        
        
        
        


