# Interaktives Finanz-Dashboard

## Projektübersicht
Dieses Projekt ist ein interaktives Finanz-Dashboard zur Analyse von Aktien und Kryptowährungen.
Es lädt historische Marktdaten, berechnet Kennzahlen und technische Indikatoren, visualisiert
Ergebnisse interaktiv und ermöglicht den Vergleich mehrerer Assets sowie die Analyse eines Portfolios.

Die Anwendung kann sowohl:
- **als Konsolenprogramm (CLI)**  
- **als interaktive Webanwendung (Dash / Plotly)**  
genutzt werden.

---

## Projektstruktur

Uni-Projekt-2025/
│
├── main.py # Einstiegspunkt (CLI oder Dashboard)
├── requirements.txt # Python-Abhängigkeiten
├── README.md # Projektbeschreibung
│
└── src/
├── data_fetch.py # Datenbeschaffung & Caching
├── analysis.py # Kennzahlen & technische Indikatoren
├── visualize.py # Erste Visualisierungen (Matplotlib)
├── forecast.py # Zeitreihenprognose (ARIMA, optional)
├── comparison.py # Vergleich mehrerer Assets
├── portfolio.py # Portfolio-Analyse (Equal Weight, FX)
└── app.py # Dash-Webanwendungteraktives Finanz-Dashboard

## Datenbeschaffung (`data_fetch.py`)
Dieses Modul ist für das Laden historischer Kursdaten zuständig.

**Funktionen und Eigenschaften:**
- Datenabruf über **Yahoo Finance** (`yfinance`)
- Unterstützung für Aktien und Kryptowährungen (z.B. `AAPL`, `MSFT`, `BTC-USD`)
- Lokales **Caching** der Daten als CSV-Dateien
- Retry-Mechanismus mit Backoff bei Netzwerkproblemen
- Fallback auf Cache-Daten bei temporären API-Ausfällen

**Zentrale Funktion:**
```python
load_data(ticker, period="1y", interval="1d")

**Vorteile:**
- Reduzierte API-Abfragen
- höhere Stabilität bei temporären Netzwerk- oder API-Problemen
- einheitliches DataFrame-Format für alle weiteren Module

## Analyse & technische Indikatoren (`analysis.py`)

Dieses Modul ist für die Berechnung finanzieller Kennzahlen und technischer Indikatoren
auf Basis historischer Kursdaten zuständig. Als Grundlage dient stets der
Schlusskurs (`Close`) eines Assets.

### Grundlegende Kennzahlen
Zur Beschreibung der Kursentwicklung werden folgende Basiskennzahlen berechnet:

- letzter verfügbarer Schlusskurs
- Höchstkurs im betrachteten Zeitraum
- Tiefstkurs im betrachteten Zeitraum
- Durchschnittskurs im Zeitraum
- Rendite über den Zeitraum (in Prozent)
- tägliche Volatilität (Standardabweichung der Tagesrenditen)

```python
compute_basic_stats(df)

### Technische Indikatoren

Zusätzlich zu den grundlegenden Kennzahlen unterstützt das Analyse-Modul gängige
technische Indikatoren aus der Finanzanalyse. Diese Indikatoren dienen der
Identifikation von Trends, Momentum und potenziellen Über- bzw. Unterbewertungen.

- **Simple Moving Average (SMA)**  
  Einfacher gleitender Durchschnitt, der den Kursverlauf über ein festes Zeitfenster
  glättet und kurzfristige Schwankungen reduziert.

- **Exponential Moving Average (EMA)**  
  Gewichteter gleitender Durchschnitt, bei dem neuere Kurse stärker berücksichtigt
  werden als ältere. Dadurch reagiert der EMA schneller auf Marktveränderungen
  als der SMA.

- **Relative Strength Index (RSI)**  
  Momentum-Indikator mit einem Wertebereich von 0 bis 100.  
  Typischerweise gelten Werte über 70 als überkauft und Werte unter 30 als überverkauft.

- **MACD (Moving Average Convergence Divergence)**  
  Kombination aus Trend- und Momentum-Indikator.  
  Besteht aus der MACD-Linie (Differenz zweier EMAs), einer Signallinie
  sowie einem Histogramm, das den Abstand zwischen beiden Linien darstellt.

Die Berechnung der Indikatoren erfolgt über folgende Funktionen:

```python
add_moving_average(df, window)
add_ema(df, span)
add_rsi(df, period=14)
add_macd(df)

## 📉 Visualisierung (`visualize.py`)

Dieses Modul stellt einfache Visualisierungsfunktionen auf Basis von **Matplotlib** bereit.
Es dient **nicht** als Ersatz für das interaktive Dashboard, sondern erfüllt einen
klar abgegrenzten, ergänzenden Zweck innerhalb des Projekts.

### Zweck und Motivation

Die Visualisierung in diesem Modul wird primär für:
- die **Konsolenanwendung (CLI)**
- schnelle visuelle Kontrolle während der Entwicklung
- Debugging und Exploration der geladenen Daten

verwendet.

Im Gegensatz zur Dash-Webanwendung:
- benötigt dieses Modul **keinen Webserver**
- funktioniert vollständig **offline**
- erlaubt eine schnelle Darstellung direkt aus der Konsole heraus


### Funktionalität

Das Modul fokussiert sich bewusst auf **preisbasierte Darstellungen** und einfache
gleitende Durchschnitte:

- Darstellung des reinen Kursverlaufs (`Close`)
- Kursverläufe mit gleitenden Durchschnitten (Moving Averages)

Die Konsolenvisualisierung dient dabei ausschließlich der explorativen Analyse
und ist nicht für eine vollständige technische Analyse vorgesehen. 
Die vollständige und interaktive Visualisierung der Analyseergebnisse erfolgt
im Web-Dashboard auf Basis von **Dash** und **Plotly**.

### Dashboard-Anwendung (`app.py`)

Die Datei `app.py` implementiert die interaktive Weboberfläche des Projekts.  
Sie basiert auf **Dash** (Web-Framework) und **Plotly** (interaktive Visualisierung) und
verbindet die einzelnen Module (`data_fetch.py`, `analysis.py`, `comparison.py`, `portfolio.py`,
optional `forecast.py`) zu einer einheitlichen Anwendung.

### Ziel des Dashboards
Das Dashboard dient als zentrale Oberfläche, um:
- einzelne Assets schnell zu analysieren (Kurs, Kennzahlen, Indikatoren)
- mehrere Assets vergleichbar darzustellen (normalisierte Verläufe, Vergleichstabelle)
- ein virtuelles Portfolio zu bewerten (Equal Weight, FX-Umrechnung, Kennzahlen-Tabelle)

Die Darstellung ist bewusst interaktiv umgesetzt (Zoom, Hover, gemeinsame X-Achse),
um Explorations- und Analyseprozesse zu unterstützen.


### Aufbau: Tabs (drei Anwendungsbereiche)

Das Dashboard ist in drei Tabs unterteilt, um Funktionalitäten klar zu trennen:

#### 1) Single Asset Analysis
- Eingabe eines einzelnen Tickers (z.B. `AAPL`, `BTC-USD`)
- Auswahl eines Zeitraums (yfinance `period`)
- optionale Overlays/Indikatoren:
  - Moving Average (MA)
  - Exponential Moving Average (EMA)
  - RSI(14)
  - MACD
  - optional: ARIMA Forecast
- Ausgabe:
  - interaktiver Kurschart (Plotly) inkl. Indikator-Panels (RSI/MACD)
  - Kennzahlen-Karten (z.B. letzter Schlusskurs, High/Low/Mean)

Technisch:
- beim Klick auf **„Daten laden“** wird ein Dash-Callback ausgelöst
- Daten werden mit `load_data(...)` geladen und bei Bedarf mit Indikatoren ergänzt
- die Visualisierung wird über Plotly Subplots aufgebaut (Preis + optionale Panels)


#### 2) Compare Assets
- Eingabe mehrerer Ticker (komma-separiert)
- Laden historischer Daten pro Asset
- Vergleich über einen einheitlichen Zeitraum (z.B. letzte 365 Tage)
- Ausgabe:
  - normalisierte Kursverläufe (Start = 100) für direkte Vergleichbarkeit
  - Kennzahlen-Tabelle (Rendite, Volatilität, Drawdown etc.)
  - Statushinweise, falls einzelne Ticker nicht geladen werden konnten

Technisch:
- nutzt die Klassen `Asset` und `Comparator` aus `comparison.py`
- Kennzahlen werden pro Asset berechnet und anschließend tabellarisch zusammengeführt


#### 3) Portfolio Analyse
- Eingabe mehrerer Ticker (komma-separiert)
- Wahl einer Basiswährung (`EUR` oder `USD`)
- automatische FX-Umrechnung aller Assets in die Basiswährung
- Equal-Weight Portfolio-Index mit täglichem Rebalancing
- Ausgabe:
  - normierte Zeitreihen (Assets + Portfolio) in einer Plotly-Figure
  - Kennzahlen-Tabelle über mehrere Zeiträume (YTD, 1Y, 3Y, 5Y)
  - Status-/Warnhinweise (z.B. fehlende FX-Daten, ausgeschlossene Ticker)

Technisch:
- nutzt Funktionen aus `portfolio.py`
  - `download_prices(...)`
  - `convert_prices_to_base_currency(...)`
  - `build_equal_weight_portfolio_index(...)`
  - `build_metrics_table(...)`
  - `build_price_figure(...)`


### Optionales Feature: ARIMA Forecast (statsmodels)
Die Zeitreihenprognose im *Single Asset Analysis*-Tab basiert auf einem ARIMA-Modell
und nutzt das optionale Python-Paket statsmodels.

Dieses Paket ist **nicht zwingend erforderlich**, um das Dashboard zu starten oder zu nutzen.

**Ohne installiertes `statsmodels`:**
- das Dashboard startet normal
- alle Tabs (Single Asset, Compare Assets, Portfolio Analyse) sind vollständig nutzbar
- lediglich die **ARIMA-Forecast-Funktion ist deaktiviert**
- beim Aktivieren des Forecasts erscheint ein entsprechender Hinweis im UI

**Mit installiertem `statsmodels`:**
- die ARIMA-Prognose kann zusätzlich im Single-Asset-Tab genutzt werden
- es werden Prognosewerte inklusive Konfidenzintervall dargestellt

***Hinweis***
Das Fehlen des Pakets statsmodels schränkt ausschließlich die Prognosefunktion ein und beeinträchtigt nicht den Betrieb des Dashboards.

### Start der Webanwendung
Das Dashboard wird über den Menüpunkt in `main.py` (Modus 2) oder direkt gestartet und läuft unter:

## Einstiegspunkt des Projekts (`main.py` im Projekt-Root)

Das Projekt wird über die Datei `main.py` im **Projekt-Root** gestartet:

```bash
python main.py

Diese Datei dient als zentraler Einstiegspunkt für das gesamte Projekt. Sie leitet den Programmfluss anschließend intern an die entsprechenden Module im src/-Verzeichnis weiter.

Die Entscheidung für einen separaten Einstiegspunkt im Projekt-Root wurde bewusst getroffen:

Der Projekt-Root fungiert als klarer Startpunkt für den Anwender, während das src/-Verzeichnis ausschließlich Implementierungslogik enthält.


Dadurch entsteht eine klare Trennung zwischen:

Anwendungsstart (Orchestrierung)
→ main.py im Root

Fachlogik und Module
→ Dateien im src/-Ordner

Weiterleitung innerhalb des Projekts

Das Root-main.py:

- verarbeitet die Benutzerauswahl (CLI oder Dashboard)

- ruft anschließend gezielt Funktionen aus src/ auf (z.B. src.app.main() für das Dashboard)

Diese Struktur ermöglicht:

- einen einheitlichen Einstiegspunkt

- einfache Erweiterbarkeit (z.B. weitere Startmodi)

- klare Verantwortlichkeiten der einzelnen Module

- Vorteil für Wartbarkeit


