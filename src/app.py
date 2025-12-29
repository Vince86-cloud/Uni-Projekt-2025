import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pandas as pd
import dash
from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go

from src.data_fetch import load_data
from src.analysis import compute_basic_stats, add_moving_average

app = Dash(__name__, title="Market Dashboard")
server = app.server


def _format_price(value):
    if value is None:
        return "-"
    try:
        return f"{float(value):,.2f}"
    except Exception:
        return "-"


def _empty_figure(title: str = "Kein Diagramm verfuegbar"):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Datum",
        yaxis_title="Schlusskurs",
    )
    return fig


def _build_figure(df, ticker: str):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df["Close"],
            mode="lines",
            name="Close",
            line=dict(width=2, color="#1f77b4"),
        )
    )

    ma_cols = [col for col in df.columns if col.startswith("MA_")]
    for col in ma_cols:
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df[col],
                mode="lines",
                name=col,
                line=dict(width=1.6),
            )
        )

    fig.update_layout(
        title=f"{ticker} Kursverlauf",
        template="plotly_white",
        xaxis_title="Datum",
        yaxis_title="Schlusskurs",
        legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="left", x=0),
        margin=dict(l=40, r=20, t=60, b=80),
        hovermode="x unified",
    )
    return fig


app.layout = html.Div(
    style={
        "fontFamily": "Segoe UI, sans-serif",
        "background": "linear-gradient(135deg, #f7f9fb 0%, #eef2f7 100%)",
        "minHeight": "100vh",
        "padding": "32px",
    },
    children=[
        html.Div(
            style={"maxWidth": "1600px", "margin": "0 auto", "background": "white", "padding": "28px", "borderRadius": "12px", "boxShadow": "0 10px 30px rgba(0,0,0,0.08)"},
            children=[
                html.H1("Finanz-Dashboard", style={"marginBottom": "6px", "fontWeight": 700}),
                html.P(
                    "Lade Kursdaten, berechne gleitende Durchschnitte und visualisiere den Verlauf interaktiv.",
                    style={"color": "#4a5568", "marginBottom": "18px"},
                ),
                html.Div(
                    style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "alignItems": "flex-end", "marginBottom": "18px"},
                    children=[
                        html.Div(
                            children=[
                                html.Label("Ticker", style={"fontWeight": 600}),
                                dcc.Input(
                                    id="ticker-input",
                                    type="text",
                                    value="AAPL",
                                    placeholder="z.B. AAPL, MSFT, BTC-USD",
                                    style={"width": "200px", "padding": "10px", "borderRadius": "8px", "border": "1px solid #cbd5e0"},
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label("MA-Fenster (Tage)", style={"fontWeight": 600}),
                                dcc.Input(
                                    id="window-input",
                                    type="number",
                                    value=20,
                                    min=1,
                                    step=1,
                                    style={"width": "160px", "padding": "10px", "borderRadius": "8px", "border": "1px solid #cbd5e0"},
                                ),
                            ]
                        ),
                        html.Div([
                            html.Label("Zeitraum", style={"fontWeight": 600}),
                            dcc.DatePickerRange(
                                id="date-range",
                                min_date_allowed="2000-01-01",
                                max_date_allowed=None,
                                start_date=None,
                                end_date=None,
                                display_format="YYYY-MM-DD",
                                style={"padding": "4px"}
                            ),
                        ]),
                        html.Button(
                            "Daten laden",
                            id="load-button",
                            n_clicks=0,
                            style={
                                "padding": "12px 18px",
                                "background": "#2563eb",
                                "color": "white",
                                "border": "none",
                                "borderRadius": "10px",
                                "fontWeight": 600,
                                "cursor": "pointer",
                                "boxShadow": "0 4px 12px rgba(37, 99, 235, 0.3)"
                            },
                        ),
                        html.Div(id="error-message", style={"color": "#c53030", "fontWeight": 600}),
                    ],
                ),
                html.Div(
                    id="stats-container",
                    style={"display": "grid", "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))", "gap": "12px", "marginBottom": "18px"},
                    children=[
                        html.Div(
                            style={"background": "#f8fafc", "padding": "14px", "borderRadius": "10px", "border": "1px solid #e2e8f0"},
                            children=[html.Div("Letzter Schlusskurs", style={"color": "#4a5568", "fontWeight": 600}), html.H3(id="latest-price", style={"margin": 0})],
                        ),
                        html.Div(
                            style={"background": "#f8fafc", "padding": "14px", "borderRadius": "10px", "border": "1px solid #e2e8f0"},
                            children=[html.Div("Hoechstkurs (1J)", style={"color": "#4a5568", "fontWeight": 600}), html.H3(id="max-price", style={"margin": 0})],
                        ),
                        html.Div(
                            style={"background": "#f8fafc", "padding": "14px", "borderRadius": "10px", "border": "1px solid #e2e8f0"},
                            children=[html.Div("Tiefstkurs (1J)", style={"color": "#4a5568", "fontWeight": 600}), html.H3(id="min-price", style={"margin": 0})],
                        ),
                        html.Div(
                            style={"background": "#f8fafc", "padding": "14px", "borderRadius": "10px", "border": "1px solid #e2e8f0"},
                            children=[html.Div("Durchschnitt (1J)", style={"color": "#4a5568", "fontWeight": 600}), html.H3(id="mean-price", style={"margin": 0})],
                        ),
                    ],
                ),
                dcc.Graph(id="price-chart", figure=_empty_figure("Bitte Ticker laden")),
            ],
        )
    ],
)


@app.callback(
    Output("price-chart", "figure"),
    Output("latest-price", "children"),
    Output("max-price", "children"),
    Output("min-price", "children"),
    Output("mean-price", "children"),
    Output("error-message", "children"),
    Input("load-button", "n_clicks"),
    State("ticker-input", "value"),
    State("window-input", "value"),
    State("date-range", "start_date"),
    State("date-range", "end_date"),
)
def update_dashboard(n_clicks, ticker, window, start_date, end_date):
    if not n_clicks:
        return _empty_figure("Bitte Ticker laden"), "-", "-", "-", "-", ""

    if not ticker:
        return _empty_figure(), "-", "-", "-", "-", "Bitte einen Ticker eingeben."

    try:
        window = int(window) if window else 20
        window = max(1, window)
    except Exception:
        window = 20

    ticker_clean = ticker.strip().upper()

    try:
        df = load_data(ticker_clean)
        if df is None or df.empty:
            raise ValueError("Keine Daten verfuegbar.")
        # Zeitraum filtern, falls gesetzt
        idx = pd.to_datetime(df.index)
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        if start_date:
            start_dt = pd.to_datetime(start_date)
            if start_dt.tzinfo is not None:
                start_dt = start_dt.tz_localize(None)
            df = df[idx >= start_dt]
        if end_date:
            end_dt = pd.to_datetime(end_date)
            if end_dt.tzinfo is not None:
                end_dt = end_dt.tz_localize(None)
            df = df[idx <= end_dt]
        if df.empty:
            raise ValueError("Keine Daten im gewählten Zeitraum.")
        df = add_moving_average(df, window=window)
        stats = compute_basic_stats(df)
        fig = _build_figure(df, ticker_clean)
    except Exception as exc:
        return _empty_figure(), "-", "-", "-", "-", f"Fehler: {exc}"

    return (
        fig,
        _format_price(stats.get("latest_price")),
        _format_price(stats.get("max_last_year")),
        _format_price(stats.get("min_last_year")),
        _format_price(stats.get("mean_last_year")),
        "",
    )


if __name__ == "__main__":
    app.run(debug=True)
