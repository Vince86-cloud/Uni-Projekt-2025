# Asset Comparison Tool
***
1. [Generelle Informationen](#generelle-informationen)
2. [Verwendete Technologien](#verwendete-technologien)
3. [Projektstruktur](#projektstruktur)
4. [Installation](#installation)
5. [Quickstart](#quickstart)
6. [Designentscheidungen](#designentscheidungen)
7. [Rollenverteilung](#rollenverteilung)
***

# Generelle Informationen
Dieses Modul dient der Berechnung zentraler Finanzkennzahlen für einzelne Finanz-Assets
über absolute Zeiträume (Start-/Enddatum) sowie relative Zeiträume (z. B. letzte 30/90/365 Tage).
Auf Basis dieser Kennzahlen ermöglicht das Modul einen strukturierten Vergleich von mindestens
zwei oder mehr Finanz-Assets.

1. Problemstellung
- Das Modul „Comparison“ löst die Problematik der Unvergleichbarkeit von Finanzkennzahlen mehrerer Finanz-Assets und ermöglicht deren Vergleichbarkeit durch die Nutzung von Bibliotheken wie pandas, NumPy und datetime sowie zur Datenbeschaffung (z.B. über yFinance).
2. Funktionsweise
- Das Modul verwendet zwei Klassen, um einen Vergleich von zwei oder mehr Assets zu realisieren. 
- Die erste Klasse "Asset" berechnet für einen Finanz-Asset zentrale Kennzahlen wie Rendite, Volatilität, Drawdown, normierte Gesamtperformance, Preisentwicklung, bester Tagesgewinn und schlechtester Tagesverlust.
- Die Kennzahlen können sowohl einzeln als auch gebündelt berechnet werden.
- Zusätzlich enthält die Klasse Hilfsmethoden zur Validierung von Datumsangaben hinsichtlich Datumsformat, Zeitzonenkonsistenz, zeitlicher Reihenfolge sowie der Verfügbarkeit von Daten im angegebenen Zeitraum.
- Zeitangaben innerhalb der Datumsparameter werden bewusst tolerant behandelt, da die Analyse auf Tagesdaten basiert und Uhrzeiten keinen Einfluss auf die Berechnung der Kennzahlen haben.
- Die zweite Klasse „Comparator“ enthält keine eigene Berechnungslogik, sondern greift auf die Klasse „Asset“ zu und nutzt deren berechnete Kennzahlen.
- Hierbei werden die Kennzahlen für den besten und schlechtesten Tag bewusst ausgeklammert, da ein Vergleich aufgrund unterschiedlicher Kalendertage nicht sinnvoll ist.
- Beim Vergleich von zwei Finanz-Assets werden absolute Differenzen sowie relative Abweichungen berechnet und tabellarisch dargestellt.
- Eine zusätzliche Spalte gibt eine Bewertung ab, welches Asset hinsichtlich der jeweiligen Kategorie besser abschneidet
- Bei mehr als zwei Finanz-Assets werden die Kennzahlen relativ zum Mittelwert verglichen und das jeweils beste Asset pro Kennzahl bestimmt.

# Verwendete Technologien 
- Programmiersprache: Python
- Bibliotheken: Numpy, Pandas
- Standardbibliotheken: datetime

# Projektstruktur
- Die Datei comparison.py enthält den gesamten Code für die Klassen zur Erstellung der Finanzkennzahlen sowie deren Vergleich
- Die Datei requirements.txt enthält alle Module die für die Erstellung des Vergleichs notwendig sind 

# Installation
- Python >= 3.10 empfohlen
- pip install -r requirements.txt

# Quickstart
- Bevor der unten gezeigte Code ausgeführt werden kann, müssen zunächst auf die Daten der Finanzkennzahlen aus yfinance zugegriffen und in Variablen gespeichert werden.
- Die DataFrames müssen mindestens eine DatetimeIndex-Struktur sowie eine "Close"-Spalte enthalten.

```python
from comparison import Asset, Comparator

asset_a = Asset("AAPL", df_aapl)
asset_b = Asset("MSFT", df_msft)

comp = Comparator([asset_a, asset_b])
result = comp.compare_metrics(
    comp.collect_metrics(days=90)
)
```

# Designentscheidungen
- Absolute und relative Zeiträume werden bewusst über eine gemeinsame API unterstützt,
  um Redundanz zu vermeiden und die Wartbarkeit zu erhöhen.
- Zeitangaben im Datum werden tolerant behandelt, da die Analyse auf Tagesdaten basiert.
- Kennzahlen wie bester/schlechtester Tag werden nicht in den Vergleich einbezogen,
  da sie aufgrund unterschiedlicher Kalendertage nicht sinnvoll vergleichbar sind.
- Das Modul ist datenquellenunabhängig gestaltet und erwartet vorbereitete pandas-DataFrames, um Wiederverwendbarkeit und Testbarkeit zu gewährleisten.

# Rollenverteilung
- Asset & Comparator Logik: Ufuk Türkkan
- Visualisierung:
- Portfolio-Modul: 


