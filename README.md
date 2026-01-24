# Interaktives Finanz-Dashboard

## Projektübersicht
Dieses Projekt ist ein interaktives Finanz-Dashboard zur Analyse von Aktien und Kryptowährungen.  
Es lädt historische Marktdaten, stellt den **zuletzt verfügbaren Kurs** eines Assets bereit, berechnet Kennzahlen und technische Indikatoren, visualisiert Ergebnisse interaktiv und ermöglicht den Vergleich mehrerer Assets sowie die Analyse eines Portfolios.

Die Anwendung kann sowohl:
- **als Konsolenprogramm (CLI)**  
- **als interaktive Webanwendung (Dash / Plotly)**  
genutzt werden.

**Hinweis:** Der „aktuelle Kurs“ entspricht dem zuletzt von der Datenquelle bereitgestellten Marktpreis  
(keine tickbasierte Echtzeitversorgung).

## Projekt starten 

1. Abhängigkeiten installieren  
     
  Zunächst sind die benötigten Pakete zu installieren:
  
   pip install -r requirements.txt
  
  Für die optionale ARIMA-Zeitreihenprognose wird das Paket `statsmodels` benötigt.
  Dieses kann bei Bedarf zusätzlich installiert werden:

    pip install -r requirements-forecast.txt

2. Projekt starten

  Das Projekt aus dem Projekt-Root starten:

    python main.py

3. Anwendungsmodus wählen

  Nach dem Start kann zwischen zwei Modi gewählt werden:

    1 → CLI (konsolenbasierte Analyse)

    2 → Dashboard (Webanwendung)

4. Dashboard aufrufen

  Bei Auswahl des Dashboard-Modus ist die Anwendung unter folgender Adresse erreichbar:

    http://127.0.0.1:8050

**Projektstruktur**

Uni-Projekt-2025/
│
├── main.py              # Einstiegspunkt (CLI oder Dashboard)
├── requirements.txt     # Verwendete Module/Bibliotheken
├── README.md            # Projektbeschreibung
│
└── src/
    ├── data_fetch.py    # Datenbeschaffung & Caching
    ├── analysis.py      # Kennzahlen & technische Indikatoren
    ├── visualize.py     # Konsolenvisualisierung (Matplotlib)
    ├── forecast.py      # Zeitreihenprognose
    ├── comparison.py    # Vergleich mehrerer Assets
    ├── portfolio.py     # Portfolio-Analyse (Equal Weight, FX)
    └── Dashboard.py     # Dash-Webanwendung

**Datenbeschaffung (data_fetch.py)**

Dieses Modul ist für das Laden historischer Kursdaten zuständig und stellt eine robuste
Schnittstelle zur externen Datenquelle bereit.

**Funktionen und Eigenschaften**

- Datenabruf über Yahoo Finance (yfinance)

- Unterstützung für Aktien und Kryptowährungen (z.B. AAPL, MSFT, BTC-USD)

- Lokales Caching der Daten als CSV-Dateien

- Retry-Mechanismus mit exponentiellem Backoff bei Netzwerkproblemen

- Fallback auf ältere Cache-Daten bei temporären API-Ausfällen

Zur Erhöhung der Stabilität wird eine **wiederverwendbare HTTP-Session** eingesetzt,
in der u.a. ein realistischer User-Agent sowie optionale SSL-Zertifikatskonfigurationen
berücksichtigt werden.

**Zentrale Funktion**

load_data(ticker, period="1y", interval="1d")

**Vorteile**

- Reduzierte API-Abfragen

- Höhere Stabilität bei temporären Netzwerk- oder API-Problemen

- Einheitliches pandas.DataFrame-Format für alle weiteren Module

- Funktioniert auch bei kurzfristigen Ausfällen der Datenquelle

**Analyse & technische Indikatoren (analysis.py)**

Dieses Modul berechnet finanzielle Kennzahlen und technische Indikatoren
auf Basis historischer Kursdaten.
Als Grundlage dient stets der Schlusskurs (Close) eines Assets.

**Grundlegende Kennzahlen**

- letzter verfügbarer Schlusskurs

- Höchst- und Tiefstkurs im betrachteten Zeitraum

- Durchschnittskurs

- Rendite (in Prozent)

- tägliche Volatilität (Standardabweichung der Tagesrenditen)

compute_basic_stats(df)

**Technische Indikatoren**

- Simple Moving Average (SMA)

- Exponential Moving Average (EMA)

- Relative Strength Index (RSI)

- MACD (Moving Average Convergence Divergence)

add_moving_average(df, window)
add_ema(df, span)
add_rsi(df, period=14)
add_macd(df)

## Zeitreihenprognose (`forecast.py`)

Dieses Modul stellt eine **optionale Zeitreihenprognose** für einzelne Assets bereit.
Es dient der exemplarischen Erweiterung des Projekts um einfache
prognostische Verfahren aus der Zeitreihenanalyse.

### Funktionalität
- Prognose zukünftiger Kursentwicklungen auf Basis historischer Schlusskurse
- Verwendung eines **ARIMA-Modells** (AutoRegressive Integrated Moving Average)
- Darstellung von Prognosewerten inklusive **Konfidenzintervallen**

### Technische Umsetzung
- Umsetzung mit dem Python-Paket **statsmodels**
- Die Prognose wird ausschließlich im **Single-Asset-Analyse-Tab** des Dashboards angeboten
- Die Berechnung erfolgt **on demand**, d.h. nur bei expliziter Aktivierung durch den Nutzer

### Optionaler Charakter
Das Modul ist bewusst als **optionales Feature** implementiert:

- Ist `statsmodels` **nicht installiert**:
  - Das Projekt startet und läuft vollständig
  - Alle Analyse- und Visualisierungsfunktionen sind nutzbar
  - Die Prognosefunktion ist deaktiviert und wird im UI entsprechend gekennzeichnet

- Ist `statsmodels` **installiert**:
  - Die ARIMA-Prognose kann zusätzlich aktiviert werden
  - Prognose und Unsicherheitsbereich werden visualisiert


**Visualisierung (visualize.py)**

Dieses Modul stellt einfache Visualisierungen auf Basis von Matplotlib bereit.
Es dient primär der Konsolenanwendung (CLI) und der explorativen Analyse.

- Kein Webserver erforderlich

- Offline nutzbar

- Schnelle visuelle Kontrolle der Kursdaten

Die Konsolenvisualisierung dient ausschließlich der explorativen Analyse
und ist nicht für eine vollständige technische Analyse vorgesehen. 
Die vollständige und interaktive Visualisierung der Analyseergebnisse erfolgt
im Web-Dashboard auf Basis von **Dash** und **Plotly**.

**Dashboard-Anwendung (Dashboard.py)**

Die Datei Dashboard.py implementiert die interaktive Weboberfläche des Projekts
auf Basis von Dash und Plotly.
Sie verbindet die Module data_fetch.py, analysis.py, comparison.py,
portfolio.py sowie forecast.py.

**Ziel des Dashboards**

- Analyse einzelner Assets

- Vergleich mehrerer Assets

- Bewertung eines virtuellen Portfolios

Die Darstellung ist bewusst interaktiv (Zoom, Hover, gemeinsame Zeitachse),
um explorative Analyseprozesse zu unterstützen.

**Aufbau: Tabs**

1) Single Asset Analyse

- Analyse eines einzelnen Tickers

- Anzeige von Kurs, Kennzahlen und Indikatoren

- Optionaler ARIMA-Forecast

2) Asset-Vergleich

- Vergleich mehrerer Assets

- Normalisierte Kursverläufe

- Vergleichstabelle mit Kennzahlen

3) Portfolio-Analyse

- Equal-Weight-Portfolio

- FX-Umrechnung in eine Basiswährung

- Kennzahlen über mehrere Zeiträume

**Optionales Feature: ARIMA-Prognose**

Die Zeitreihenprognose basiert auf einem ARIMA-Modell (statsmodels) und ist optional.

**Ohne installiertes statsmodels:**

- Dashboard vollständig nutzbar

- Prognosefunktion deaktiviert

**Mit installiertem statsmodels:**

- zusätzliche Prognose inkl. Konfidenzintervall im Single-Asset-Tab

***Hinweis***
Das Fehlen des Pakets statsmodels schränkt ausschließlich die Prognosefunktion ein und beeinträchtigt nicht den Betrieb des Dashboards.

**Einstiegspunkt des Projekts (main.py)**

Das Projekt wird über die Datei main.py im Projekt-Root gestartet:
python main.py


Die Datei main.py fungiert als zentraler Einstiegspunkt und steuert,
ob die Anwendung im CLI-Modus oder als Web-Dashboard gestartet wird.

Diese Struktur ermöglicht:

- einen klaren Einstiegspunkt für Nutzer

- eine saubere Trennung von **Steuerungslogik (Programmfluss)** und **Fachlogik (Analyse, Berechnung, Visualisierung)**

- gute Wartbarkeit und einfache Erweiterbarkeit


