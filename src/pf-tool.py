"""
Portfolio-Kennzahlen-Dashboard (Dash + yfinance) – Equal Weight + Graph + FX-Umrechnung

Ziel:
- Nutzer gibt mehrere Ticker ein (komma-separiert)
- App lädt historische Daten via yfinance (auto_adjust=True)
- App konvertiert alle Asset-Preisreihen in eine Basiswährung (EUR oder USD)
- App berechnet Kennzahlen (Return, Volatilität, Max Drawdown) für:
  * jedes Asset (in Basiswährung)
  * das Equal-Weight-Portfolio (in Basiswährung)
- App visualisiert im Chart normierte Verläufe (Start=100), ebenfalls in Basiswährung

Warum FX-Umrechnung nötig ist:
- Ein Portfolio mischt Werte (Preise/Renditen) nur sinnvoll, wenn alle Assets in derselben
  Währung bewertet werden. Sonst entstehen methodische Fehler.

FX-Umrechnungsprinzip:
- Wenn wir eine FX-Quote "BASEQUOTE=X" haben (z.B. EURUSD=X),
  dann bedeutet der Kurs: QUOTE pro 1 BASE.
  Beispiel EURUSD = 1.10 => 1 EUR = 1.10 USD

Umrechnung:
- Preis in QUOTE -> BASE:  Preis_BASE = Preis_QUOTE / (BASEQUOTE)
- Preis in BASE  -> QUOTE: Preis_QUOTE = Preis_BASE * (BASEQUOTE)

Wir versuchen FX-Paare automatisch zu finden:
1) bevorzugt: {base}{from}=X (z.B. EURUSD=X, wenn from=USD und base=EUR)
2) falls nicht vorhanden: {from}{base}=X und dann invertierte Rechenregel
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import numpy as np
import pandas as pd
import yfinance as yf

from dash import Dash, dcc, html, dash_table, Input, Output, State
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
        # Ein Ticker => normales DF
        close = data[["Close"]].copy()
        close.columns = tickers

    return close.dropna(how="all")


# =============================================================================
# 3) FX / Währungslogik
# =============================================================================

# Kleines In-Memory-Cache, damit yfinance.info nicht bei jedem Klick unnötig oft abgefragt wird.
_CURRENCY_CACHE: dict[str, str | None] = {}


def get_ticker_currency(ticker: str) -> str | None:
    """
    Bestimmt die Handelswährung eines Tickers über yfinance (Ticker.info["currency"]).

    Warum try/except?
    - yfinance.info kann gelegentlich fehlschlagen (Rate limits, Netzwerk, fehlende Felder).
    - Für ein robustes Dashboard sollte das abgefangen werden.

    Caching:
    - Wir speichern Ergebnisse in _CURRENCY_CACHE, um wiederholte Abfragen zu reduzieren.
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
    - base_currency: "EUR" oder "USD" (im UI auswählbar)
    - years_back: Historie, für die FX-Reihen geladen werden

    Ausgaben:
    - converted_prices: DataFrame in Basiswährung (nur konvertierbare Ticker)
    - currencies: dict {ticker: currency} zur Transparenz/Statusanzeige
    - warnings: Liste von Hinweisen, z.B. wenn ein Ticker ausgeschlossen wurde

    Designentscheidung:
    - Wenn ein Ticker nicht konvertierbar ist (Währung unbekannt oder FX fehlt),
      wird er aus converted_prices entfernt, damit Portfolio/Chart nicht „falsch“ werden.
    """
    if prices.empty:
        return prices, {}, ["Keine Preisdaten vorhanden."]

    base_currency = base_currency.upper().strip()
    warnings: list[str] = []

    # Handelswährungen bestimmen
    currencies: dict[str, str | None] = {t: get_ticker_currency(t) for t in prices.columns}

    # Cache für FX-Serien, damit ein FX-Ticker nur einmal geladen wird.
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
            warnings.append(
                f"{ticker}: FX-Rate für {ccy}->{base_currency} nicht verfügbar -> ausgeschlossen."
            )
            continue

        # Datumsindex angleichen und fehlende FX-Werte per ffill auffüllen
        fx_aligned = fx_series.reindex(prices.index).ffill()

        # Umrechnung je nach gefundenem FX-Paar
        if direct_mode:
            # fx_ticker = base+from; fx = from pro 1 base => price_base = price_from / fx
            converted[ticker] = prices[ticker] / fx_aligned
        else:
            # fx_ticker = from+base; fx = base pro 1 from => price_base = price_from * fx
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
    - Portfolio-Rendite je Tag: Durchschnitt über Assets (mean(axis=1))

    Umgang mit fehlenden Daten:
    - Wir ffillen vor Renditeberechnung, um Handelskalender-Differenzen abzufangen.
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
    - größter prozentualer Rückgang vom bisherigen Hoch.

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
    - sonst: letzte N Jahre (DateOffset)
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
    Erstellt einen DataFrame für die Tabelle:
    - Zeilen: Assets (Ticker + Portfolio)
    - Spalten: Kennzahlen je Zeitraum
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

            row[f"{spec.label} Return"] = m["return"]
            row[f"{spec.label} Volatility"] = m["vol"]
            row[f"{spec.label} Max Drawdown"] = m["mdd"]

        rows.append(row)

    return pd.DataFrame(rows)


# =============================================================================
# 6) Formatierung für die Anzeige
# =============================================================================

def format_percent(x: float) -> str:
    """0.1234 -> '12.34%'; NaN -> '—'"""
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "—"
    return f"{x * 100:,.2f}%"


def make_table_records(df: pd.DataFrame) -> tuple[list[dict], list[dict]]:
    """
    Wandelt DataFrame in Dash-DataTable-Format:
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
# 7) Chart-Erstellung
# =============================================================================

def normalize_to_100(series: pd.Series) -> pd.Series:
    """
    Normiert eine Zeitreihe auf Startwert 100:
    norm_t = series_t / series_0 * 100

    Damit lassen sich Verläufe unterschiedlicher Assets visuell vergleichen.
    """
    s = series.dropna()
    if s.shape[0] < 2:
        return pd.Series(dtype=float)
    return (s / s.iloc[0]) * 100.0


def build_price_figure(prices_base: pd.DataFrame, portfolio_index: pd.Series, base_currency: str) -> go.Figure:
    """
    Baut die Plotly-Figure für den Chart.

    Inhalt:
    - Alle Asset-Zeitreihen (in Basiswährung), normiert auf 100
    - Portfolio-Index, ebenfalls normiert auf 100
    """
    fig = go.Figure()

    for ticker in prices_base.columns:
        s_norm = normalize_to_100(prices_base[ticker])
        if not s_norm.empty:
            fig.add_trace(go.Scatter(x=s_norm.index, y=s_norm.values, mode="lines", name=ticker))

    p_norm = normalize_to_100(portfolio_index)
    if not p_norm.empty:
        fig.add_trace(go.Scatter(
            x=p_norm.index,
            y=p_norm.values,
            mode="lines",
            name=portfolio_index.name,
            line=dict(width=4)
        ))

    fig.update_layout(
        title=f"Kursverläufe (Start=100) in Basiswährung {base_currency}",
        xaxis_title="Datum",
        yaxis_title=f"Index (Start=100, bewertet in {base_currency})",
        hovermode="x unified",
        legend_title="Asset",
        margin=dict(l=40, r=20, t=60, b=40),
    )
    return fig


# =============================================================================
# 8) Dash App
# =============================================================================

app = Dash(__name__)
server = app.server

app.layout = html.Div(
    style={"maxWidth": "1250px", "margin": "40px auto", "fontFamily": "Arial"},
    children=[
        html.H2("Portfolio-Kennzahlen (yfinance + Dash) – Equal Weight + Graph + FX"),

        html.Div(
            style={"marginBottom": "10px", "display": "flex", "gap": "14px", "alignItems": "center"},
            children=[
                html.Div(children=[
                    html.Label("Ticker (komma-separiert):"),
                    dcc.Input(
                        id="ticker-input",
                        type="text",
                        value="AAPL,MSFT,GLD",
                        style={"width": "420px", "marginLeft": "10px"},
                    ),
                ]),
                html.Div(children=[
                    html.Label("Basiswährung:"),
                    dcc.Dropdown(
                        id="base-currency",
                        options=[
                            {"label": "EUR", "value": "EUR"},
                            {"label": "USD", "value": "USD"},
                        ],
                        value="EUR",
                        clearable=False,
                        style={"width": "140px"},
                    ),
                ]),
                html.Button("Aktualisieren", id="btn-run", n_clicks=0),
            ],
        ),

        html.Div(
            style={"marginBottom": "14px", "color": "#444"},
            children=[
                html.Small(
                    "Hinweis: Portfolio und Kennzahlen werden nach FX-Umrechnung in der Basiswährung berechnet. "
                    "Volatilität ist annualisiert (√252). Portfolio = Equal Weight mit täglichem Rebalancing."
                )
            ],
        ),

        html.Div(id="status-text", style={"marginBottom": "10px", "color": "#b00"}),

        dcc.Graph(id="price-graph", figure=go.Figure(), style={"height": "520px"}),

        html.H3("Kennzahlen-Tabelle", style={"marginTop": "24px"}),

        dash_table.DataTable(
            id="metrics-table",
            data=[],
            columns=[],
            style_table={"overflowX": "auto"},
            style_cell={"padding": "8px", "textAlign": "left", "whiteSpace": "normal"},
            style_header={"fontWeight": "bold"},
        ),
    ],
)


@app.callback(
    Output("metrics-table", "data"),
    Output("metrics-table", "columns"),
    Output("price-graph", "figure"),
    Output("status-text", "children"),
    Input("btn-run", "n_clicks"),
    State("ticker-input", "value"),
    State("base-currency", "value"),
)
def update_dashboard(n_clicks: int, raw_tickers: str, base_currency: str):
    """
    Callback-Ablauf:
    1) Ticker parsen
    2) Rohpreise laden (je Asset in Handelswährung)
    3) Preise in Basiswährung konvertieren (FX-Logik)
    4) Portfolio-Index berechnen (Equal Weight) in Basiswährung
    5) Chart erstellen
    6) Kennzahlen-Tabelle erstellen (Assets + Portfolio)
    7) Status-/Warnhinweise ausgeben
    """
    tickers = parse_tickers(raw_tickers)

    if not tickers:
        return [], [], go.Figure(), "Bitte mindestens einen Ticker eingeben (z.B. AAPL oder AAPL,MSFT)."

    prices_raw = download_prices(tickers, years_back=6.0)
    if prices_raw.empty:
        return [], [], go.Figure(), "Keine Daten erhalten. Prüfe Ticker-Symbole und Internetverbindung."

    # Hinweis: yfinance liefert evtl. nicht für alle gewünschten Ticker Daten
    available_raw = set(prices_raw.columns)
    missing_raw = [t for t in tickers if t not in available_raw]

    # FX-Konvertierung: nur konvertierbare Assets bleiben übrig
    prices_base, currencies, fx_warnings = convert_prices_to_base_currency(
        prices_raw, base_currency=base_currency, years_back=6.0
    )

    if prices_base.empty:
        status_lines = []
        if missing_raw:
            status_lines.append(f"Keine Preisdaten für: {', '.join(missing_raw)}")
        status_lines.extend(fx_warnings)
        return [], [], go.Figure(), " | ".join(status_lines)

    # Portfolio und Chart nur auf Basiswährung
    portfolio_index = build_equal_weight_portfolio_index(prices_base, base_value=100.0)
    fig = build_price_figure(prices_base, portfolio_index, base_currency=base_currency)

    # Tabelle: Assets + Portfolio
    combined = prices_base.copy()
    combined[portfolio_index.name] = portfolio_index

    metrics_df = build_metrics_table(combined)
    records, columns = make_table_records(metrics_df)

    # Statusmeldungen zusammensetzen
    status_lines: list[str] = []
    if missing_raw:
        status_lines.append(f"Keine Preisdaten für: {', '.join(missing_raw)}")

    # Zeige zusätzlich Währungsinfo (hilfreich für Dozent)
    # (nur kurz – sonst wird es zu lang)
    used_assets = list(prices_base.columns)
    status_lines.append(f"Verwendete Assets (in {base_currency}): {', '.join(used_assets)}")

    # FX-Warnungen
    status_lines.extend(fx_warnings)

    return records, columns, fig, " | ".join(status_lines)


if __name__ == "__main__":
    app.run(debug=True)
