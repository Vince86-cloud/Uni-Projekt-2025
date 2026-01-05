from __future__ import annotations

from typing import Optional

from dash import Dash, html, dcc, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd

from .data_fetch import load_data
from .analysis import (
    compute_basic_stats,
    add_moving_average,
    add_ema,
    add_rsi,
    add_macd,
)

# Forecast (ARIMA) – optional dependency statsmodels (handled with try/except)
try:
    from .forecast import arima_forecast, pick_arima_order_quick

    _FORECAST_AVAILABLE = True
except Exception:
    arima_forecast = None
    pick_arima_order_quick = None
    _FORECAST_AVAILABLE = False


app = Dash(__name__, title="Market Dashboard")
server = app.server


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


def _empty_figure(title: str = "Kein Diagramm verfügbar"):
    fig = go.Figure()
    fig.update_layout(
        title=title,
        template="plotly_white",
        xaxis_title="Datum",
        yaxis_title="Schlusskurs",
        hovermode="x unified",
    )
    return fig


def _is_crypto_ticker(ticker: str) -> bool:
    # simple heuristic: Yahoo crypto tickers often end with "-USD"
    t = (ticker or "").upper()
    return t.endswith("-USD") or t.endswith("-EUR") or t.endswith("-USDT")


def _align_forecast_index(
    forecast_df: pd.DataFrame,
    last_date: pd.Timestamp,
    steps: int,
    crypto: bool,
) -> pd.DataFrame:
    """
    Our forecast module uses business days by default. For crypto we prefer daily.
    This function reindexes the forecast output to the desired calendar.
    """
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

    # For stocks business days are OK (keep as-is)
    return forecast_df


def _build_figure(
    df: pd.DataFrame,
    ticker: str,
    overlays: list[str],
    forecast_df: Optional[pd.DataFrame] = None,
):
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

    # Panel 1: Close
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

    # Forecast on price panel (optional)
    if show_forecast:
        # Confidence band
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

    # MA: all MA_* columns, if active
    if "MA" in overlays:
        ma_cols = sorted([c for c in df.columns if c.startswith("MA_")])
        for col in ma_cols:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=col.replace("MA_", "MA(") + ")",
                    line=dict(width=1.6),
                ),
                row=1,
                col=1,
            )

    # EMA: selected
    ema_map = {
        "EMA20": ("EMA_20", "EMA(20)"),
        "EMA50": ("EMA_50", "EMA(50)"),
        "EMA100": ("EMA_100", "EMA(100)"),
    }
    for key, (col, name) in ema_map.items():
        if key in overlays and col in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=df[col],
                    mode="lines",
                    name=name,
                    line=dict(width=1.6),
                ),
                row=1,
                col=1,
            )

    current_row = 2

    # RSI Panel
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

    # MACD Panel
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


app.layout = html.Div(
    style={
        "fontFamily": "Segoe UI, sans-serif",
        "background": "linear-gradient(135deg, #f7f9fb 0%, #eef2f7 100%)",
        "minHeight": "100vh",
        "padding": "32px",
    },
    children=[
        html.Div(
            style={
                "maxWidth": "1600px",
                "margin": "0 auto",
                "background": "white",
                "padding": "28px",
                "borderRadius": "12px",
                "boxShadow": "0 10px 30px rgba(0,0,0,0.08)",
            },
            children=[
                html.H1("Finanz-Dashboard", style={"marginBottom": "6px", "fontWeight": 700}),
                html.P(
                    "Lade Kursdaten, berechne Indikatoren (MA/EMA/RSI/MACD) und visualisiere den Verlauf interaktiv. "
                    "Optional: ARIMA-Prognose mit Konfidenzintervall.",
                    style={"color": "#4a5568", "marginBottom": "18px"},
                ),
                html.Div(
                    style={
                        "display": "flex",
                        "gap": "12px",
                        "flexWrap": "wrap",
                        "alignItems": "flex-end",
                        "marginBottom": "10px",
                    },
                    children=[
                        html.Div(
                            children=[
                                html.Label("Ticker", style={"fontWeight": 600}),
                                dcc.Input(
                                    id="ticker-input",
                                    type="text",
                                    value="AAPL",
                                    placeholder="z.B. AAPL, MSFT, BTC-USD",
                                    style={
                                        "width": "200px",
                                        "padding": "10px",
                                        "borderRadius": "8px",
                                        "border": "1px solid #cbd5e0",
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label("Zeitraum", style={"fontWeight": 600}),
                                dcc.Dropdown(
                                    id="period-input",
                                    options=[
                                        {"label": p, "value": p}
                                        for p in ["1mo", "3mo", "6mo", "1y", "2y", "5y", "max"]
                                    ],
                                    value="1y",
                                    clearable=False,
                                    style={"width": "160px"},
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
                                    style={
                                        "width": "160px",
                                        "padding": "10px",
                                        "borderRadius": "8px",
                                        "border": "1px solid #cbd5e0",
                                    },
                                ),
                            ]
                        ),
                        html.Div(
                            children=[
                                html.Label("Forecast-Horizont (Tage)", style={"fontWeight": 600}),
                                dcc.Input(
                                    id="forecast-steps",
                                    type="number",
                                    value=30,
                                    min=1,
                                    step=1,
                                    style={
                                        "width": "200px",
                                        "padding": "10px",
                                        "borderRadius": "8px",
                                        "border": "1px solid #cbd5e0",
                                    },
                                ),
                            ]
                        ),
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
                                "boxShadow": "0 4px 12px rgba(37, 99, 235, 0.3)",
                            },
                        ),
                        html.Div(id="error-message", style={"color": "#c53030", "fontWeight": 600}),
                    ],
                ),
                html.Div(
                    style={"marginBottom": "16px"},
                    children=[
                        html.Label("Indikatoren anzeigen", style={"fontWeight": 600}),
                        dcc.Checklist(
                            id="indicator-toggle",
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
                            style={"marginTop": "6px"},
                        ),
                        html.Div(
                            "Hinweis: ARIMA Forecast benötigt das optionale Paket 'statsmodels'. "
                            "Falls nicht installiert, wird eine Meldung angezeigt und der Forecast nicht gezeichnet.",
                            style={"color": "#4a5568", "marginTop": "6px"},
                        ),
                    ],
                ),
                html.Div(
                    id="stats-container",
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "repeat(auto-fit, minmax(180px, 1fr))",
                        "gap": "12px",
                        "marginBottom": "18px",
                    },
                    children=[
                        _stat_card("Letzter Schlusskurs", "latest-price"),
                        _stat_card("Höchstkurs (Zeitraum)", "max-price"),
                        _stat_card("Tiefstkurs (Zeitraum)", "min-price"),
                        _stat_card("Durchschnitt (Zeitraum)", "mean-price"),
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
    State("period-input", "value"),
    State("window-input", "value"),
    State("forecast-steps", "value"),
    State("indicator-toggle", "value"),
)
def update_dashboard(n_clicks, ticker, period, window, forecast_steps, overlays):
    if not n_clicks:
        return _empty_figure("Bitte Ticker laden"), "-", "-", "-", "-", ""

    if not ticker:
        return _empty_figure(), "-", "-", "-", "-", "Bitte einen Ticker eingeben."

    overlays = overlays or []

    # sanitize inputs
    try:
        window = int(window) if window else 20
        window = max(1, window)
    except Exception:
        window = 20

    try:
        forecast_steps = int(forecast_steps) if forecast_steps else 30
        forecast_steps = max(1, forecast_steps)
        forecast_steps = min(forecast_steps, 180)  # keep UI responsive
    except Exception:
        forecast_steps = 30

    ticker_clean = ticker.strip().upper()
    period = (period or "1y").strip()

    error_msg = ""

    try:
        df = load_data(ticker_clean, period=period, interval="1d")
        if df is None or df.empty:
            raise ValueError("Keine Daten verfügbar.")

        # Indicators on demand
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

        # Forecast (optional) – never crash the whole app
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

        fig = _build_figure(df, ticker_clean, overlays, forecast_df=forecast_df)

    except Exception as exc:
        return _empty_figure(), "-", "-", "-", "-", f"Fehler: {exc}"

    return (
        fig,
        _format_price(stats.get("latest_price")),
        _format_price(stats.get("max_last_year")),
        _format_price(stats.get("min_last_year")),
        _format_price(stats.get("mean_last_year")),
        error_msg,
    )


if __name__ == "__main__":
    app.run(debug=True)