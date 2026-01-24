from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from dash import Dash, html, dcc, Input, Output, State, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Single-Asset Module ---
from src.data_fetch import load_data
from src.analysis import (
    compute_basic_stats,
    add_moving_average,
    add_ema,
    add_rsi,
    add_macd,
)

# --- Compare-Assets Module ---
from src.comparison import Asset, Comparator

# --- Portfolio Module (refactored aus pf-tool) ---
from src.portfolio import (
    parse_tickers as pf_parse_tickers,
    download_prices as pf_download_prices,
    convert_prices_to_base_currency,
    build_equal_weight_portfolio_index,
    build_metrics_table,
    make_table_records,
    build_price_figure,
)

# --- Forecast (ARIMA) optional ---
try:
    from src.forecast import arima_forecast, pick_arima_order_quick

    _FORECAST_AVAILABLE = True
except Exception:
    arima_forecast = None
    pick_arima_order_quick = None
    _FORECAST_AVAILABLE = False


app = Dash(__name__, title="Finanz-Dashboard", suppress_callback_exceptions=True)
server = app.server
POPULAR_TICKERS = [
    {"label": "Apple (AAPL)", "value": "AAPL"},
    {"label": "Microsoft (MSFT)", "value": "MSFT"},
    {"label": "Alphabet (GOOGL)", "value": "GOOGL"},
    {"label": "Amazon (AMZN)", "value": "AMZN"},
    {"label": "Tesla (TSLA)", "value": "TSLA"},
    {"label": "Bitcoin (BTC-USD)", "value": "BTC-USD"},
    {"label": "Ethereum (ETH-USD)", "value": "ETH-USD"},
]
SOURCE_LABELS = {
    "live": "Live-Daten (Yahoo Finance)",
    "cache": "Lokaler Cache",
    "stale_cache": "Älterer Cache (Fallback)",
}

# =============================================================================
# Helper
# =============================================================================

def _format_price(value) -> str:
    if value is None:
        return "-"
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "-"


def _stat_card(title: str, element_id: str):
    return html.Div(
        style={
            "background": "#f8fafc",
            "padding": "14px",
            "borderRadius": "10px",
            "border": "1px solid #e2e8f0",
        },
        children=[
            html.Div(title, style={"color": "#4a5568", "fontWeight": 600}),
            html.H3(id=element_id, style={"margin": 0}),
        ],
    )


def _empty_figure(title: str = "Kein Diagramm verfügbar") -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Datum",
        yaxis_title="Wert",
        hovermode="x unified",
    )
    return fig


def _is_crypto_ticker(ticker: str) -> bool:
    t = (ticker or "").upper()
    return t.endswith("-USD") or t.endswith("-EUR") or t.endswith("-USDT")


def _align_forecast_index(
    forecast_df: pd.DataFrame,
    last_date: pd.Timestamp,
    steps: int,
    crypto: bool,
) -> pd.DataFrame:
    if forecast_df is None or forecast_df.empty:
        return forecast_df

    if crypto:
        future_index = pd.date_range(
            start=last_date + pd.Timedelta(days=1),
            periods=steps,
            freq="D",
        )
        out = forecast_df.copy()
        out.index = future_index
        return out

    return forecast_df


def _build_single_asset_figure(
    df: pd.DataFrame,
    ticker: str,
    overlays: list[str],
    forecast_df: Optional[pd.DataFrame] = None,
) -> go.Figure:
    overlays = overlays or []

    show_rsi = ("RSI14" in overlays) and ("RSI_14" in df.columns)
    show_macd = ("MACD" in overlays) and all(
        c in df.columns for c in ["MACD", "MACD_Signal", "MACD_Hist"]
    )
    show_forecast = (forecast_df is not None) and (not forecast_df.empty)

    rows = 1 + (1 if show_rsi else 0) + (1 if show_macd else 0)
    row_heights = [1.0] if rows == 1 else ([0.7, 0.3] if rows == 2 else [0.6, 0.2, 0.2])

    fig = make_subplots(
        rows=rows,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        row_heights=row_heights,
    )

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(width=2),
        ),
        row=1,
        col=1,
    )

    if show_forecast:
        x = list(forecast_df.index) + list(forecast_df.index[::-1])
        y = list(forecast_df["yhat_upper"]) + list(forecast_df["yhat_lower"][::-1])

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                fill="toself",
                name="Forecast CI (95%)",
                hoverinfo="skip",
                line=dict(width=0),
                opacity=0.2,
            ),
            row=1,
            col=1,
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_df.index,
                y=forecast_df["yhat"],
                mode="lines",
                name="ARIMA Forecast",
                line=dict(width=2, dash="dash"),
            ),
            row=1,
            col=1,
        )

    if "MA" in overlays:
        ma_cols = sorted([c for c in df.columns if c.startswith("MA_")])
        for col in ma_cols:
            fig.add_trace(
                go.Scatter(x=df.index, y=df[col], mode="lines", name=col, line=dict(width=1.6)),
                row=1,
                col=1,
            )

    ema_map = {"EMA20": "EMA_20", "EMA50": "EMA_50", "EMA100": "EMA_100"}
    for key, col in ema_map.items():
        if key in overlays and col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col.replace("_", "(") + ")",
                    line=dict(width=1.6),
                ),
                row=1,
                col=1,
            )

    current_row = 2

    if show_rsi:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["RSI_14"],
                mode="lines",
                name="RSI(14)",
                line=dict(width=1.6),
            ),
            row=current_row,
            col=1,
        )
        fig.add_hline(y=70, line_dash="dash", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", row=current_row, col=1)
        fig.update_yaxes(range=[0, 100], title_text="RSI", row=current_row, col=1)
        current_row += 1

    if show_macd:
        fig.add_trace(
            go.Bar(x=df.index, y=df["MACD_Hist"], name="MACD Hist", opacity=0.5),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=df.index, y=df["MACD"], mode="lines", name="MACD", line=dict(width=1.6)),
            row=current_row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df["MACD_Signal"],
                mode="lines",
                name="Signal",
                line=dict(width=1.6),
            ),
            row=current_row,
            col=1,
        )
        fig.add_hline(y=0, line_dash="dash", row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", row=current_row, col=1)

    fig.update_layout(
        title=f"{ticker} – Kurs & Indikatoren",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=90),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Preis", row=1, col=1)
    return fig


# =============================================================================
# Layouts für Tabs
# =============================================================================

def _base_container(children):
    return html.Div(
        style={
            "maxWidth": "1250px",
            "margin": "40px auto",
            "fontFamily": "Arial",
        },
        children=children,
    )


def single_asset_layout():
    return html.Div(
        children=[
            html.H2("Single Asset Analysis"),

            html.Div(
                style={
                    "marginBottom": "10px",
                    "display": "flex",
                    "gap": "14px",
                    "alignItems": "center",
                    "flexWrap": "wrap",
                },
                children=[
                    html.Div(children=[
                        html.Label("Ticker:", style={"fontWeight": 600}),
                        html.Div(
                            style={"display": "flex", "gap": "8px", "alignItems": "center", "marginLeft": "0px"},
                            children=[
                                dcc.Dropdown(
                                    id="single-ticker-dd",
                                    options=POPULAR_TICKERS,
                                    value="AAPL",
                                    placeholder="Populär auswählen",
                                    searchable=True,
                                    clearable=True,
                                    style={"width": "200px"},
                                ),
                                dcc.Input(
                                    id="single-ticker-custom",
                                    type="text",
                                    placeholder="oder eigener Ticker (z.B. SAP.DE)",
                                    style={"width": "220px"},
                                ),
                            ],
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("Zeitraum:", style={"fontWeight": 600}),
                        dcc.Dropdown(
                            id="single-period",
                            options=[{"label": p, "value": p} for p in ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]],
                            value="1y",
                            clearable=False,
                            style={"width": "200px"},
                            className="period-dd", 
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("MA-Fenster (Tage):", style={"fontWeight": 600}),
                        dcc.Input(id="single-window", type="number", value=20, min=1, step=1, style={"width": "140px"}),
                    ]),
                    html.Div(children=[
                        html.Label("Forecast (Tage):", style={"fontWeight": 600}),
                        dcc.Input(id="single-forecast-steps", type="number", value=30, min=1, step=1, style={"width": "140px"}),
                    ]),
                    html.Button("Daten laden", id="single-load", n_clicks=0),
                    html.Div(id="single-error", style={"color": "#b00", "fontWeight": 600}),
                    html.Div(
                        id="single-source",
                        style={"color": "#444", "marginTop": "6px", "fontStyle": "italic"},
                    ),],),

            html.Div(
                style={"marginBottom": "12px"},
                children=[
                    html.Label("Indikatoren anzeigen:", style={"fontWeight": 600}),
                    dcc.Checklist(
                        id="single-indicators",
                        options=[
                            {"label": "MA", "value": "MA"},
                            {"label": "EMA(20)", "value": "EMA20"},
                            {"label": "EMA(50)", "value": "EMA50"},
                            {"label": "EMA(100)", "value": "EMA100"},
                            {"label": "RSI(14)", "value": "RSI14"},
                            {"label": "MACD", "value": "MACD"},
                            {"label": "ARIMA Forecast", "value": "FORECAST"},
                        ],
                        value=["MA", "EMA100"],
                        inline=True,
                    ),
                    html.Div(
                        "Hinweis: ARIMA Forecast benötigt das optionale Paket 'statsmodels'.",
                        style={"color": "#444", "marginTop": "6px"},
                    ),
                ],
            ),

            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                    "gap": "12px",
                    "marginBottom": "16px",
                },
                children=[
                    _stat_card("Letzter Schlusskurs", "single-latest"),
                    _stat_card("Höchstkurs (Zeitraum)", "single-max"),
                    _stat_card("Tiefstkurs (Zeitraum)", "single-min"),
                    _stat_card("Durchschnitt (Zeitraum)", "single-mean"),
                ],
            ),

            dcc.Graph(id="single-chart", figure=_empty_figure("Bitte Ticker laden"), style={"height": "520px"}),
        ]
    )


def compare_assets_layout():
    return html.Div(
        children=[
            html.H2("Compare Assets"),

            html.Div(
                style={"marginBottom": "10px", "display": "flex", "gap": "14px", "alignItems": "flex-end", "flexWrap": "wrap"},
                children=[
                    html.Div(children=[
                        html.Label("Ticker (komma-separiert):", style={"fontWeight": 600}),
                        dcc.Input(
                            id="cmp-tickers",
                            type="text",
                            value="AAPL,MSFT",
                            style={"width": "420px", "marginLeft": "10px"},
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("Zeitraum (yfinance):", style={"fontWeight": 600}),
                        dcc.Dropdown(
                            id="cmp-period",
                            options=[{"label": p, "value": p} for p in ["3mo", "6mo", "1y", "2y", "5y", "max"]],
                            value="1y",
                            clearable=False,
                            style={"width": "160px"},
                            className="period-dd",  # ✅ optional (macht’s konsistent)
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("Vergleich (letzte Tage):", style={"fontWeight": 600}),
                        dcc.Dropdown(
                            id="cmp-days",
                            options=[{"label": str(x), "value": x} for x in [30, 90, 180, 365, 730]],
                            value=365,
                            clearable=False,
                            style={"width": "160px"},
                            className="period-dd",  # ✅ optional
                        ),
                    ]),
                    html.Button("Vergleichen", id="cmp-run", n_clicks=0),
                    html.Div(id="cmp-status", style={"color": "#b00", "fontWeight": 600}),
                ],
            ),

            dcc.Graph(id="cmp-chart", figure=_empty_figure("Noch kein Vergleich"), style={"height": "520px"}),

            html.H3("Kennzahlen-Vergleich", style={"marginTop": "18px"}),

            dash_table.DataTable(
                id="cmp-table",
                data=[],
                columns=[],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "textAlign": "left", "whiteSpace": "normal"},
                style_header={"fontWeight": "bold"},
            ),
        ]
    )


def portfolio_layout():
    return html.Div(
        children=[
            html.H2("Portfolio Analyse"),

            html.Div(
                style={"marginBottom": "10px", "display": "flex", "gap": "14px", "alignItems": "flex-end", "flexWrap": "wrap"},
                children=[
                    html.Div(children=[
                        html.Label("Ticker (komma-separiert):", style={"fontWeight": 600}),
                        dcc.Input(
                            id="pf-tickers",
                            type="text",
                            value="AAPL,MSFT,GLD",
                            style={"width": "420px", "marginLeft": "10px"},
                        ),
                    ]),
                    html.Div(children=[
                        html.Label("Basiswährung:", style={"fontWeight": 600}),
                        dcc.Dropdown(
                            id="pf-base",
                            options=[{"label": "EUR", "value": "EUR"}, {"label": "USD", "value": "USD"}],
                            value="EUR",
                            clearable=False,
                            style={"width": "140px"},
                            className="period-dd",  # ✅ optional
                        ),
                    ]),
                    html.Button("Aktualisieren", id="pf-run", n_clicks=0),
                    html.Div(id="pf-status", style={"color": "#b00", "fontWeight": 600}),
                ],
            ),

            html.Div(
                style={"marginBottom": "14px", "color": "#444"},
                children=[
                    html.Small(
                        "Hinweis: Portfolio/Kennzahlen werden nach FX-Umrechnung in der Basiswährung berechnet. "
                        "Volatilität ist annualisiert (√252). Portfolio = Equal Weight mit täglichem Rebalancing."
                    )
                ],
            ),

            dcc.Graph(id="pf-chart", figure=_empty_figure("Noch kein Portfolio"), style={"height": "520px"}),

            html.H3("Kennzahlen-Tabelle", style={"marginTop": "18px"}),

            dash_table.DataTable(
                id="pf-table",
                data=[],
                columns=[],
                style_table={"overflowX": "auto"},
                style_cell={"padding": "8px", "textAlign": "left", "whiteSpace": "normal"},
                style_header={"fontWeight": "bold"},
            ),
        ]
    )


# =============================================================================
# Main Layout mit Tabs
# =============================================================================

app.layout = _base_container(
    [
        html.H1("Finanz-Dashboard", style={"marginBottom": "6px"}),
        html.Div(
            "Tabs: Single Asset Analysis • Compare Assets • Portfolio Analyse",
            style={"color": "#555", "marginBottom": "12px"},
        ),

        dcc.Tabs(
            id="tabs",
            value="tab-single",
            children=[
                dcc.Tab(label="Single Asset Analysis", value="tab-single"),
                dcc.Tab(label="Compare Assets", value="tab-compare"),
                dcc.Tab(label="Portfolio Analyse", value="tab-portfolio"),
            ],
        ),

        html.Div(id="tab-content", style={"marginTop": "16px"}),
    ]
)


@app.callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab_value: str):
    if tab_value == "tab-compare":
        return compare_assets_layout()
    if tab_value == "tab-portfolio":
        return portfolio_layout()
    return single_asset_layout()


# =============================================================================
# Callback: Single Asset Analysis
# =============================================================================

@app.callback(
    Output("single-chart", "figure"),
    Output("single-latest", "children"),
    Output("single-max", "children"),
    Output("single-min", "children"),
    Output("single-mean", "children"),
    Output("single-error", "children"),
    Output("single-source", "children"),
    Input("single-load", "n_clicks"),
    State("single-ticker-dd", "value"),
    State("single-ticker-custom", "value"),
    State("single-period", "value"),
    State("single-window", "value"),
    State("single-forecast-steps", "value"),
    State("single-indicators", "value"),
)
def update_single_asset(n_clicks, ticker_dd, ticker_custom, period, window, forecast_steps, overlays):
    if not n_clicks:
        return _empty_figure("Bitte Ticker laden"), "-", "-", "-", "-", "",""
    ticker = (ticker_custom or "").strip() or (ticker_dd or "")
    if not ticker:
        return _empty_figure(), "-", "-", "-", "-", "Bitte einen Ticker eingeben.",""

    overlays = overlays or []

    try:
        window = int(window) if window else 20
        window = max(1, window)
    except Exception:
        window = 20

    try:
        forecast_steps = int(forecast_steps) if forecast_steps else 30
        forecast_steps = max(1, min(forecast_steps, 180))
    except Exception:
        forecast_steps = 30

    ticker_clean = ticker.strip().upper()
    period = (period or "1y").strip()

    error_msg = ""

    try:
        df, source = load_data(ticker_clean, period=period, interval="1d")
        if df is None or df.empty:
            raise ValueError("Keine Daten verfügbar.")

        if "MA" in overlays:
            df = add_moving_average(df, window=window)
        if "EMA20" in overlays:
            df = add_ema(df, 20)
        if "EMA50" in overlays:
            df = add_ema(df, 50)
        if "EMA100" in overlays:
            df = add_ema(df, 100)
        if "RSI14" in overlays:
            df = add_rsi(df, 14)
        if "MACD" in overlays:
            df = add_macd(df)

        stats = compute_basic_stats(df)

        forecast_df = None
        if "FORECAST" in overlays:
            if not _FORECAST_AVAILABLE:
                error_msg = "ARIMA nicht verfügbar: Bitte 'statsmodels' installieren (pip install statsmodels)."
            else:
                try:
                    order = pick_arima_order_quick(df)
                    forecast_df = arima_forecast(df, steps=forecast_steps, order=order)

                    last_date = pd.to_datetime(df.index[-1])
                    forecast_df = _align_forecast_index(
                        forecast_df,
                        last_date=last_date,
                        steps=forecast_steps,
                        crypto=_is_crypto_ticker(ticker_clean),
                    )
                except Exception as exc:
                    error_msg = f"Forecast konnte nicht berechnet werden: {exc}"
                    forecast_df = None

        fig = _build_single_asset_figure(df, ticker_clean, overlays, forecast_df=forecast_df)

        source_label = SOURCE_LABELS.get(source, source)
        return (
            fig,
            _format_price(stats.get("latest_price")),
            _format_price(stats.get("max_last_year")),
            _format_price(stats.get("min_last_year")),
            _format_price(stats.get("mean_last_year")),
            error_msg,
            f"Datenquelle: {source_label}",
        )

    except Exception as exc:
        return _empty_figure(), "-", "-", "-", "-", f"Fehler: {exc}", ""


# =============================================================================
# Callback: Compare Assets
# =============================================================================
@app.callback(
    Output("cmp-chart", "figure"),
    Output("cmp-table", "data"),
    Output("cmp-table", "columns"),
    Output("cmp-status", "children"),
    Input("cmp-run", "n_clicks"),
    State("cmp-tickers", "value"),
    State("cmp-period", "value"),
    State("cmp-days", "value"),
)
def update_compare(n_clicks, raw, period, days):
    if not n_clicks:
        return _empty_figure("Noch kein Vergleich"), [], [], ""

    try:
        tickers = [t.strip().upper() for t in (raw or "").split(",") if t.strip()]
        if len(tickers) < 2:
            return _empty_figure(), [], [], "Bitte mindestens 2 Ticker angeben (z.B. AAPL,MSFT)."

        period = (period or "1y").strip()
        days = int(days) if days else 365

        assets: list[Asset] = []
        missing: list[str] = []
        sources: dict[str, str] = {}

        for t in tickers:
            try:
                df, source = load_data(t, period=period, interval="1d")
                if df is None or df.empty:
                    missing.append(t)
                    continue

                # Sicherstellen: Close vorhanden
                if "Close" not in df.columns:
                    raise ValueError(f"{t}: Spalte 'Close' fehlt")

                # Sicherstellen: DatetimeIndex
                # Index robust auf "timezone-naive datetime64[ns]" normalisieren
                idx = pd.to_datetime(df.index, utc=True, errors="coerce")  # <- WICHTIG: utc=True
                # falls irgendwas nicht konvertierbar ist, rausfiltern
                mask = ~pd.isna(idx)
                df = df.loc[mask].copy()
                df.index = pd.DatetimeIndex(idx[mask]).tz_convert(None)  # UTC -> naive
                df = df.sort_index()
                assets.append(Asset(t, df))
                sources[t] = SOURCE_LABELS.get(source, source)

            except Exception as e:
                missing.append(f"{t} ({e})")

        if len(assets) < 2:
            return _empty_figure(), [], [], f"Zu wenige gültige Assets. Fehlend: {', '.join(missing)}"

        comp = Comparator(assets)

        # --- Kennzahlen ---
        metrics = comp.collect_metrics(days=days)
        table_df = comp.compare_metrics(metrics)

        table_df = table_df.reset_index().rename(columns={"index": "Metric"})
        table_df.columns = [str(c) for c in table_df.columns]

        # JSON-safe machen
        def _to_jsonable(x):
            if isinstance(x, (np.floating, np.integer)):
                return x.item()
            if isinstance(x, pd.Timestamp):
                return x.isoformat()
            return x

        table_df = table_df.map(_to_jsonable)

        columns = [{"name": c, "id": c} for c in table_df.columns]
        data = table_df.to_dict("records")

        # --- Chart ---
        fig = go.Figure()
        for a in assets:
            s = a.normalized_price_series(days=days)
            if s is not None and not s.empty:
                fig.add_trace(
                    go.Scatter(x=s.index, y=s.values, mode="lines", name=a.ticker)
                )

        fig.update_layout(
            title=f"Normalisierte Kursverläufe (Start=100) – letzte {days} Tage",
            template="plotly_white",
            hovermode="x unified",
            xaxis_title="Datum",
            yaxis_title="Index (Start=100)",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        status_parts = []

        if sources:
            src_text = ", ".join([f"{t}: {s}" for t, s in sources.items()])
            status_parts.append("Datenquellen: " + src_text)

        if missing:
            status_parts.append("Nicht geladen/fehlerhaft: " + ", ".join(missing))
            
        status = " | ".join(status_parts)   

        return fig, data, columns, status

    # vollständigen Traceback im UI anzeigen
    except Exception:
        import traceback
        return (
            _empty_figure("Fehler im Compare-Callback"),
            [],
            [],
            traceback.format_exc(),
        )


# =============================================================================
# Callback: Portfolio Analyse
# =============================================================================

@app.callback(
    Output("pf-chart", "figure"),
    Output("pf-table", "data"),
    Output("pf-table", "columns"),
    Output("pf-status", "children"),
    Input("pf-run", "n_clicks"),
    State("pf-tickers", "value"),
    State("pf-base", "value"),
)
def update_portfolio(n_clicks, raw_tickers, base_currency):
    if not n_clicks:
        return _empty_figure("Noch kein Portfolio"), [], [], ""

    tickers = pf_parse_tickers(raw_tickers)
    if not tickers:
        return _empty_figure(), [], [], "Bitte mindestens einen Ticker eingeben."

    prices_raw = pf_download_prices(tickers, years_back=6.0)
    if prices_raw.empty:
        return _empty_figure(), [], [], "Keine Daten erhalten. Prüfe Ticker-Symbole und Internetverbindung."

    available_raw = set(prices_raw.columns)
    missing_raw = [t for t in tickers if t not in available_raw]

    prices_base, currencies, warnings = convert_prices_to_base_currency(
        prices_raw,
        base_currency=base_currency,
        years_back=6.0,
    )

    if prices_base.empty:
        status_lines = []
        if missing_raw:
            status_lines.append(f"Keine Preisdaten für: {', '.join(missing_raw)}")
        status_lines.extend(warnings)
        return _empty_figure(), [], [], " | ".join(status_lines)

    portfolio_index = build_equal_weight_portfolio_index(prices_base, base_value=100.0)

    fig = build_price_figure(prices_base, portfolio_index, base_currency=base_currency)

    combined = prices_base.copy()
    combined[portfolio_index.name] = portfolio_index

    metrics_df = build_metrics_table(combined)
    records, columns = make_table_records(metrics_df)

    status_lines: list[str] = []
    if missing_raw:
        status_lines.append(f"Keine Preisdaten für: {', '.join(missing_raw)}")

    status_lines.append(f"Verwendete Assets (in {base_currency}): {', '.join(list(prices_base.columns))}")
    status_lines.extend(warnings)
    status_lines.append("Datenquelle: yfinance (Portfolio-Download)")

    return fig, records, columns, " | ".join(status_lines)


def main():
    app.run(debug=True, use_reloader=False, host="127.0.0.1", port=8050)


if __name__ == "__main__":
    main()