"""
Portfolio-Analyse-Modul (yfinance) – Equal Weight + FX-Umrechnung

Ziel:
- Dieses Modul stellt Funktionen zur Portfolio-Analyse bereit und ist für die
  Integration in ein übergeordnetes Dashboard (z.B. Dash-App) gedacht.
- Nutzer können mehrere Ticker (komma-separiert) analysieren.
- Historische Kursdaten werden über yfinance geladen (auto_adjust=True).
- Alle Asset-Preisreihen werden in eine gemeinsame Basiswährung
  (z.B. EUR oder USD) umgerechnet.
- Es werden Kennzahlen (Return, Volatilität, Max Drawdown) berechnet für:
  * jedes einzelne Asset (in Basiswährung)
  * ein Equal-Weight-Portfolio (in Basiswährung)
- Zusätzlich können normierte Kursverläufe (Start=100) für Assets und Portfolio
  visualisiert werden.

Warum FX-Umrechnung nötig ist:
- Ein Portfolio kombiniert Werte (Preise und Renditen) nur dann sinnvoll,
  wenn alle Assets in derselben Währung bewertet werden.
- Ohne FX-Umrechnung entstehen methodische Verzerrungen und Fehlinterpretationen.

FX-Umrechnungsprinzip:
- Wenn eine FX-Quote "BASEQUOTE=X" vorliegt (z.B. EURUSD=X),
  bedeutet der Kurs: QUOTE pro 1 BASE.
  Beispiel:
    EURUSD = 1.10  =>  1 EUR = 1.10 USD

Umrechnung:
- Preis in QUOTE -> BASE:
    Preis_BASE = Preis_QUOTE / (BASEQUOTE)
- Preis in BASE -> QUOTE:
    Preis_QUOTE = Preis_BASE * (BASEQUOTE)

Automatische FX-Paar-Erkennung:
1) Bevorzugt: {base}{from}=X
   (z.B. EURUSD=X, wenn from=USD und base=EUR)
2) Falls nicht verfügbar: {from}{base}=X
   → Umrechnung erfolgt mit invertierter Rechenregel
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go


# =============================================================================
# 1) Perioden-Definition
# =============================================================================

@dataclass(frozen=True)
class PeriodSpec:
    """
    PeriodSpec beschreibt einen Zeitraum, für den Kennzahlen berechnet werden.

    - label: Anzeigename (z.B. "1Y")
    - years:
        * None => YTD (Year-to-Date)
        * Zahl => letzte N Jahre
    """
    label: str
    years: float | None


PERIODS = [
    PeriodSpec("YTD", None),
    PeriodSpec("1Y", 1.0),
    PeriodSpec("3Y", 3.0),
    PeriodSpec("5Y", 5.0),
]


# =============================================================================
# 2) Eingabe-Parsing / Datenabruf
# =============================================================================

def parse_tickers(raw: str) -> list[str]:
    """
    Zerlegt einen Text wie "AAPL, MSFT, GLD" in eine Liste ["AAPL","MSFT","GLD"].

    Schritte:
    - split(',') => Trennung
    - strip() => Leerzeichen entfernen
    - upper() => Standardisieren
    - leere Einträge entfernen
    """
    if not raw:
        return []
    tickers = [t.strip().upper() for t in raw.split(",")]
    return [t for t in tickers if t]


def download_prices(tickers: list[str], years_back: float = 6.0) -> pd.DataFrame:
    """
    Lädt historische Kursdaten über yfinance.

    Parameter:
    - tickers: Liste von Ticker-Symbolen
    - years_back: wir laden etwas mehr als 5 Jahre, um die 5Y-Periode robust zu haben

    Wichtig:
    - auto_adjust=True:
      => "Close" ist i.d.R. adjustiert (Splits/Dividenden berücksichtigt),
         was für Performance-Berechnungen sinnvoll ist.

    Rückgabe:
    - DataFrame mit DatetimeIndex
    - Spalten = Ticker
    - Werte = adjustierter Schlusskurs
    """
    if not tickers:
        return pd.DataFrame()

    period_str = f"{int(np.ceil(years_back))}y"

    data = yf.download(
        tickers=tickers,
        period=period_str,
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if data.empty:
        return pd.DataFrame()

    # Mehrere Ticker => MultiIndex ("Close","AAPL"), ...
    if isinstance(data.columns, pd.MultiIndex):
        close = data["Close"].copy()
    else:
        # Ein Ticker => normales DF; Spaltenname auf Ticker setzen
        close = data[["Close"]].copy()
        close.columns = tickers

    return close.dropna(how="all")


# =============================================================================
# 3) FX / Währungslogik
# =============================================================================

# In-Memory-Cache, damit yfinance.info nicht bei jedem Aufruf unnötig oft abgefragt wird
_CURRENCY_CACHE: dict[str, str | None] = {}


def get_ticker_currency(ticker: str) -> str | None:
    """
    Bestimmt die Handelswährung eines Tickers über yfinance (Ticker.info["currency"]).

    Robustheit:
    - yfinance.info kann gelegentlich fehlschlagen (Rate limits, Netzwerk, fehlende Felder).
      Daher try/except.

    Performance:
    - Ergebnisse werden gecached (in _CURRENCY_CACHE), um wiederholte Abfragen zu reduzieren.
    """
    if ticker in _CURRENCY_CACHE:
        return _CURRENCY_CACHE[ticker]

    try:
        info = yf.Ticker(ticker).info
        ccy = info.get("currency")
    except Exception:
        ccy = None

    _CURRENCY_CACHE[ticker] = ccy
    return ccy


def download_single_series(ticker: str, years_back: float = 6.0) -> pd.Series:
    """
    Lädt eine einzelne Zeitreihe (Close, auto_adjust=True) und gibt sie als Series zurück.

    Zweck:
    - wird für FX-Zeitreihen genutzt (z.B. EURUSD=X)
    """
    df = download_prices([ticker], years_back=years_back)
    if df.empty or ticker not in df.columns:
        return pd.Series(dtype=float)
    return df[ticker].dropna()


def get_fx_series(from_ccy: str, to_ccy: str, years_back: float = 6.0) -> tuple[pd.Series, str, bool]:
    """
    Liefert eine FX-Zeitreihe, um von from_ccy nach to_ccy konvertieren zu können.

    Rückgabe:
    - fx_series: Zeitreihe des FX-Kurses
    - fx_ticker_used: das verwendete yfinance FX-Symbol (z.B. "EURUSD=X")
    - direct_mode:
        * True  => fx_ticker ist "to_ccy + from_ccy + =X" (BASE=to, QUOTE=from)
                 und Umrechnung ist: price_to = price_from / fx
        * False => fx_ticker ist "from_ccy + to_ccy + =X" (BASE=from, QUOTE=to)
                 und Umrechnung ist: price_to = price_from * fx

    Logik:
    1) Versuche zunächst "to_ccyfrom_ccy=X" (z.B. EURUSD=X, wenn to=EUR und from=USD)
       => Interpretation: from pro 1 to (USD pro 1 EUR)
       => von USD nach EUR: USD / (USD pro EUR) = EUR
    2) Wenn nicht verfügbar: versuche "from_ccyto_ccy=X"
       => Interpretation: to pro 1 from (EUR pro 1 USD)
       => von USD nach EUR: USD * (EUR pro USD) = EUR
    """
    if from_ccy == to_ccy:
        return pd.Series(dtype=float), "", True

    # (1) bevorzugt: to + from (z.B. EURUSD=X)
    fx1 = f"{to_ccy}{from_ccy}=X"
    s1 = download_single_series(fx1, years_back=years_back)
    if not s1.empty:
        return s1, fx1, True

    # (2) alternative: from + to (z.B. USDEUR=X)
    fx2 = f"{from_ccy}{to_ccy}=X"
    s2 = download_single_series(fx2, years_back=years_back)
    if not s2.empty:
        return s2, fx2, False

    return pd.Series(dtype=float), "", True


def convert_prices_to_base_currency(
    prices: pd.DataFrame,
    base_currency: str,
    years_back: float = 6.0,
) -> tuple[pd.DataFrame, dict[str, str | None], list[str]]:
    """
    Konvertiert alle Asset-Preisreihen in eine Basiswährung.

    Eingaben:
    - prices: DataFrame mit Spalten=ticker und Preisen in jeweiliger Handelswährung
    - base_currency: "EUR" oder "USD"
    - years_back: Historie, für die FX-Reihen geladen werden

    Ausgaben:
    - converted_prices: DataFrame in Basiswährung (nur konvertierbare Ticker)
    - currencies: dict {ticker: currency} zur Transparenz/Statusanzeige
    - warnings: Liste von Hinweisen (z.B. ausgeschlossene Ticker)

    Designentscheidung:
    - Wenn ein Ticker nicht konvertierbar ist (Währung unbekannt oder FX fehlt),
      wird er ausgeschlossen, um methodische Fehler zu vermeiden.
    """
    if prices.empty:
        return prices, {}, ["Keine Preisdaten vorhanden."]

    base_currency = base_currency.upper().strip()
    warnings: list[str] = []

    # Handelswährungen bestimmen
    currencies: dict[str, str | None] = {t: get_ticker_currency(t) for t in prices.columns}

    # Cache für FX-Serien, damit ein FX-Ticker nur einmal geladen wird
    fx_cache: dict[tuple[str, str], tuple[pd.Series, str, bool]] = {}

    converted = pd.DataFrame(index=prices.index)

    for ticker in prices.columns:
        ccy = currencies.get(ticker)

        if ccy is None:
            warnings.append(f"{ticker}: Währung konnte nicht bestimmt werden -> ausgeschlossen.")
            continue

        ccy = ccy.upper().strip()

        # Wenn bereits Basiswährung: direkt übernehmen
        if ccy == base_currency:
            converted[ticker] = prices[ticker]
            continue

        # FX-Serie bestimmen
        key = (ccy, base_currency)  # von ccy -> base
        if key not in fx_cache:
            fx_series, fx_ticker_used, direct_mode = get_fx_series(
                from_ccy=ccy, to_ccy=base_currency, years_back=years_back
            )
            fx_cache[key] = (fx_series, fx_ticker_used, direct_mode)

        fx_series, fx_ticker_used, direct_mode = fx_cache[key]

        if fx_series.empty:
            warnings.append(f"{ticker}: FX-Rate für {ccy}->{base_currency} nicht verfügbar -> ausgeschlossen.")
            continue

        # Index angleichen und fehlende FX-Werte per ffill auffüllen
        fx_aligned = fx_series.reindex(prices.index).ffill()

        # Umrechnung je nach gefundenem FX-Paar
        if direct_mode:
            # fx = from pro 1 base => price_base = price_from / fx
            converted[ticker] = prices[ticker] / fx_aligned
        else:
            # fx = base pro 1 from => price_base = price_from * fx
            converted[ticker] = prices[ticker] * fx_aligned

    # Entferne Tage, an denen alle konvertierten Assets NaN sind
    converted = converted.dropna(how="all")

    if converted.shape[1] == 0:
        warnings.append("Keine Assets konnten in die Basiswährung umgerechnet werden.")
    else:
        warnings.append(f"Umrechnung aktiv: Alle Kennzahlen/Charts in {base_currency}.")

    return converted, currencies, warnings


# =============================================================================
# 4) Portfolio-Berechnung (Equal Weight)
# =============================================================================

def build_equal_weight_portfolio_index(price_df: pd.DataFrame, base_value: float = 100.0) -> pd.Series:
    """
    Erstellt einen Portfolio-Index (Startwert base_value, z.B. 100) in Basiswährung.

    Equal Weight (mit täglichem Rebalancing):
    - tägliche Rendite je Asset: pct_change
    - Portfolio-Rendite je Tag: Durchschnitt über Assets

    Umgang mit fehlenden Daten:
    - ffill vor Renditeberechnung, um Handelskalender-Differenzen abzufangen
    """
    if price_df.empty or price_df.shape[1] == 0:
        return pd.Series(dtype=float)

    prices_ffill = price_df.sort_index().ffill()
    returns = prices_ffill.pct_change()

    portfolio_returns = returns.mean(axis=1, skipna=True).dropna()
    portfolio_index = (1.0 + portfolio_returns).cumprod() * base_value
    portfolio_index.name = "PORTFOLIO (Equal Weight)"
    return portfolio_index


# =============================================================================
# 5) Kennzahlen-Berechnung
# =============================================================================

def compute_max_drawdown(price_series: pd.Series) -> float:
    """
    Maximum Drawdown:
    - größter prozentualer Rückgang vom bisherigen Hoch

    drawdown_t = price_t / cummax(price)_t - 1
    MDD = min(drawdown_t)
    """
    if price_series.empty:
        return np.nan
    running_max = price_series.cummax()
    drawdown = price_series / running_max - 1.0
    return float(drawdown.min())


def compute_metrics_for_period(prices: pd.Series) -> dict[str, float]:
    """
    Berechnet Return, annualisierte Volatilität und Max Drawdown für eine Zeitreihe.

    - Return: (Ende/Start) - 1
    - Volatilität: std(tägliche Renditen) * sqrt(252)
    - Max Drawdown: siehe compute_max_drawdown
    """
    if prices.dropna().shape[0] < 2:
        return {"return": np.nan, "vol": np.nan, "mdd": np.nan}

    p = prices.dropna()
    total_return = float(p.iloc[-1] / p.iloc[0] - 1.0)

    daily_returns = p.pct_change().dropna()
    vol_annual = float(daily_returns.std(ddof=1) * np.sqrt(252))

    mdd = compute_max_drawdown(p)
    return {"return": total_return, "vol": vol_annual, "mdd": mdd}


def slice_period(prices: pd.Series, years: float | None) -> pd.Series:
    """
    Schneidet eine Zeitreihe auf:
    - YTD: ab 01.01. des aktuellen Jahres (years=None)
    - sonst: letzte N Jahre
    """
    if prices.empty:
        return prices

    if not isinstance(prices.index, pd.DatetimeIndex):
        prices = prices.copy()
        prices.index = pd.to_datetime(prices.index)

    if years is None:
        start = pd.Timestamp(date.today().year, 1, 1)
    else:
        start = pd.Timestamp.today() - pd.DateOffset(years=int(years))

    return prices.loc[prices.index >= start]


def build_metrics_table(price_df_with_portfolio: pd.DataFrame) -> pd.DataFrame:
    """
    Erstellt einen DataFrame für die Kennzahlen-Tabelle:
    - Zeilen: Assets (Ticker + Portfolio)
    - Spalten: Kennzahlen je Zeitraum (YTD/1Y/3Y/5Y)
    """
    if price_df_with_portfolio.empty:
        return pd.DataFrame()

    rows = []
    for name in price_df_with_portfolio.columns:
        s = price_df_with_portfolio[name].dropna()
        row = {"Asset": name}

        for spec in PERIODS:
            sliced = slice_period(s, spec.years)
            m = compute_metrics_for_period(sliced)

            row[f"{spec.label} Rendite"] = m["return"]
            row[f"{spec.label} Volatilität"] = m["vol"]
            row[f"{spec.label} Max Drawdown"] = m["mdd"]

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 6) Formatierung für die Anzeige (Dash-DataTable)
# =============================================================================

def format_percent(x: float) -> str:
    """0.1234 -> '12.34%'; NaN -> '—'"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:,.2f}%"


def make_table_records(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Wandelt DataFrame in Dash-DataTable-Format um:
    - records: list[dict]
    - columns: list[dict]
    """
    if df.empty:
        return [], []

    display = df.copy()
    for col in display.columns:
        if col == "Asset":
            continue
        display[col] = display[col].apply(format_percent)

    columns = [{"name": c, "id": c} for c in display.columns]
    records = display.to_dict("records")
    return records, columns


# =============================================================================
# 7) Chart-Erstellung (Plotly)
# =============================================================================

def normalize_to_100(series: pd.Series) -> pd.Series:
    """
    Normiert eine Zeitreihe auf Startwert 100:
    norm_t = series_t / series_0 * 100
    """
    s = series.dropna()
    if s.shape[0] < 2:
        return pd.Series(dtype=float)
    return (s / s.iloc[0]) * 100.0


def build_price_figure(prices_base: pd.DataFrame, portfolio_index: pd.Series, base_currency: str) -> go.Figure:
    """
    Baut die Plotly-Figure für den Chart:
    - Asset-Zeitreihen (Basiswährung), normiert auf 100
    - Portfolio-Index, ebenfalls normiert auf 100
    """
    fig = go.Figure()

    for ticker in prices_base.columns:
        s_norm = normalize_to_100(prices_base[ticker])
        if not s_norm.empty:
            fig.add_trace(go.Scatter(x=s_norm.index, y=s_norm.values, mode="lines", name=ticker))

    p_norm = normalize_to_100(portfolio_index)
    if not p_norm.empty:
        fig.add_trace(
            go.Scatter(
                x=p_norm.index,
                y=p_norm.values,
                mode="lines",
                name=portfolio_index.name,
                line=dict(width=4),
            )
        )

    fig.update_layout(
        title=f"Kursverläufe (Start=100) in Basiswährung {base_currency}",
        xaxis_title="Datum",
        yaxis_title=f"Index (Start=100, bewertet in {base_currency})",
        hovermode="x unified",
        legend_title="Asset",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig
