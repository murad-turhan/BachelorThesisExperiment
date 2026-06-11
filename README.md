# Modell-Experimente zur Bachelorarbeit
**Reparaturkostenprognose bei deinmotorschaden.de unter Small-Data-Bedingungen**

Dieser Ordner enthält den vollständigen, reproduzierbaren Code für die empirischen
Experimente (Kapitel 6 *Implementierung* und Kapitel 7 *Empirische Ergebnisse*).
Alles ist exakt nach der Methodik der Arbeit (Kapitel 5) umgesetzt.

> **Stand der Ausführung:** Datenaufbereitung, stratifizierte Pseudo-Label-Auswahl
> und die deskriptive Analyse wurden bereits real ausgeführt (Ergebnisse liegen in
> `data/`, `figures/`, `results/descriptive_summary.csv`). Die drei Modellansätze
> (scikit-learn / XGBoost) müssen lokal ausgeführt werden – siehe
> [Abschnitt „Ausführung"](#ausführung-im-terminal).

---

## 1. Mapping zur Bachelorarbeit

| Datei | Kapitel | Inhalt |
|---|---|---|
| `src/data_prep.py` | 4.2–4.4 / 5.2 | Labeled-Set + Unlabeled-Pool aus Salesforce-Exporten |
| `src/pseudo_labels.py` | 5.5 | Stratifizierte Auswahl der 5.000 Pseudo-Label-Kandidaten |
| `src/preprocessing.py` | 5.2 | StandardScaler (numerisch) + OneHotEncoding (kategorial) |
| `src/models.py` | 5.3 / 6.3 | Ridge, RandomForest, XGBoost + Hyperparameter-Gitter |
| `src/metrics.py` | 5.6 | MAE, RMSE, R², MAPE |
| `src/nested_cv.py` | 5.6 | Nested CV (äußerer 5-fold, innerer 3-fold) |
| `src/approach1_baseline.py` | 5.3 / 7.2 | Ansatz 1: Baseline-Regression |
| `src/approach2_delta.py` | 5.4 / 7.3 | Ansatz 2: Delta-/Residual-Modellierung |
| `src/approach3_ssl.py` | 5.5 / 7.4 | Ansatz 3: SSL mit Pseudo-Labeling |
| `src/descriptive.py` | 7.1 | Deskriptive Analyse + Abbildungen |
| `run_all.py` | 6/7 | Orchestriert die gesamte Pipeline |

---

## 2. Projektstruktur

```
kfz-repair-pricing-experiments/
├── config.py                 # zentrale Pfade & Parameter (Seed, Folds, ...)
├── requirements.txt
├── run_all.py                # End-to-End-Lauf
├── data/                     # erzeugte Datensätze (CSV)
│   ├── labeled.csv               # 112 gelabelte Fälle (Ansatz 1 & 2)
│   ├── unlabeled_pool_full.csv   # 38.649 KVs mit Features (Ansatz 3 Pool)
│   └── pseudo_pool_5000.csv      # 5.000 stratifiziert gezogene Kandidaten
├── results/                  # Nested-CV-Ergebnisse als CSV
├── figures/                  # Abbildungen für Kap. 7.1
└── src/                      # Module (s. Tabelle oben)
```

---

## 3. Datengrundlage

Drei Salesforce-Objekte werden über relationale IDs verknüpft (Kap. 4.2). Wichtig:
Der **Offers-Export nutzt 18-stellige, der Customer-Export 15-stellige Lead-IDs** –
gejoint wird auf den ersten 15 Zeichen.

**Labeled-Set (Ansatz 1 & 2): `labeled.csv` — 112 Fälle**
Fälle mit von einer der 7 Werkstätten bestätigtem finalem Reparaturpreis.

**Unlabeled-Pool (Ansatz 3): `unlabeled_pool_full.csv` — 38.649 KVs**
`offers_with_estimates.xlsx` (alle Kostenvoranschläge, ~39 k) ⋈ vollständiger
`Customer`-Export (12.251 Leads mit Fahrzeug-Features) auf der Lead-ID; gelabelte
Leads werden ausgeschlossen → echter ungelabelter Pool.

### Feature-Selektion (Kap. 5.2)
Es werden die **zwölf strukturierten Features** der Thesis (Tab. tab:features)
verwendet:

- **Numerisch:** `year_of_construction`, `mileage`, `kw`, `hsn`, `vehicle_age`
- **Kategorial:** `brand`, `model`, `fuel_type`, `car_from_country`,
  `vehicle_ready_to_drive`, `damage_type`, `motor_code`
- **Kostenvoranschlag (`price_estimation`):** erst ab **Ansatz 2** ein Feature
  (in Ansatz 1 bewusst ausgeschlossen, vgl. Kap. 5.2).

Numerische Features: Median-Imputation + `StandardScaler`. Kategoriale Features:
Konstant-Imputation + `OneHotEncoder(handle_unknown="ignore")`. Wegen der
hochkardinalen Felder (`model`, `motor_code`) treten in Test-Folds ungesehene
Ausprägungen auf; `handle_unknown="ignore"` fängt diese ab (deshalb kein
`drop="first"`, da nicht kombinierbar). Alle Schritte werden fold-intern gefittet.

---

## 4. Die drei Ansätze

**Ansatz 1 – Baseline-Regression (Kap. 5.3).** Direkte Regression
`final_price ~ Fahrzeug-/Schadensmerkmale` (ohne KV) mit Ridge / RandomForest /
XGBoost.

**Ansatz 2 – Delta-/Residual-Modellierung (Kap. 5.4).** Zielvariable ist
`delta = final_price − KV`; vorhergesagt wird `final_price_hat = KV + delta_hat`.
Bewertung auf dem rekonstruierten Preis → direkt mit Ansatz 1 vergleichbar.
Theoretische Basis: Crane & Crotty (1967), Malpezzi (1999).
**Empirischer Beleg aus den Daten:** die Delta-Transformation halbiert die zu
lernende Varianz (Var(final) ≈ 19,0 Mio. → Var(delta) ≈ 9,2 Mio.).

**Ansatz 3 – SSL mit Pseudo-Labeling (Kap. 5.5).** Self-Training mit
Confidence-Schwellwert (Lee 2013; Rizve et al. 2021):
1. RandomForest auf gelabeltem Trainings-Fold tunen (innerer 3-fold).
2. Vorhersage `final_price` auf den 5.000 KVs; **Unsicherheit = Std. über die RF-Bäume**.
3. Die sichersten **q %** (niedrigste Std.) werden als Pseudo-Labels akzeptiert.
4. Akzeptierte ins Training übernehmen, neu fitten — **max. 3 Iterationen**.

Der Schwellwert **q ∈ {20, 30, 40 %}** wird im **inneren CV mitoptimiert** (wie die
RF-Hyperparameter). **Leakage-Schutz:** Pseudo-Labels entstehen ausschließlich aus
dem äußeren Trainings-Fold; der Test-Fold bleibt unangetastet. Zur Performance-
Degradation-Frage (van Engelen & Hoos 2020) wird je Fold zusätzlich ein **rein
überwachter** RandomForest bewertet. Die Iterations-/Schwellwert-Dynamik liefert
`src/approach3_ssl_diagnostics.py` (Kap. 7.4).

---

## 5. Validierung (Kap. 5.6)

Nested Cross-Validation: äußerer `KFold(5, shuffle, seed=42)` zur unverzerrten
Performanzschätzung, innerer `GridSearchCV(cv=3)` zur Hyperparameterwahl. Die
gesamte Vorverarbeitung steckt in der sklearn-Pipeline und wird pro Fold neu
gefittet (kein Data Leakage). Begründung der Strategie bei n = 112: Vabalas et al.
(2019).

---

## 6. Stratifizierte Pseudo-Label-Auswahl – Ergebnis

Der Pool ist BMW-/Opel-lastig, die Labels sind Mercedes-/VW-lastig. Es wird daher
auf die **Markenverteilung der echten Labels** abgestimmt (Distribution-Matching),
um den Covariate-Shift zu minimieren (stützt die SSL-Cluster-Annahme). Ergebnis:

| Marke | labeled % | gezogen % | Δ pp |
|---|---|---|---|
| Mercedes-Benz | 24,3 | 24,4 | +0,1 |
| Volkswagen | 22,5 | 22,6 | +0,1 |
| Audi | 17,1 | 17,2 | +0,1 |
| BMW | 13,5 | 13,7 | +0,2 |
| Ford | 8,1 | 8,1 | 0,0 |

KV-Preis-Median: labeled 5.994 € vs. gezogen 6.000 € → gut abgestimmt.

---

## 7. Quellen-Mapping (Methodik-Fidelität)

| Quelle | Verwendung im Code |
|---|---|
| Lee (2013) *Pseudo-Label* | Self-Training-Grundprinzip (Ansatz 3) |
| Rizve et al. (2021) *Uncertainty-Aware PL Selection* | Confidence-Schwellwert über Unsicherheit |
| van Engelen & Hoos (2020) *SSL Survey* | überwachte Referenz / Performance-Degradation-Test |
| Kim et al. (2023) *Self-Training for Tabular* | SSL-Self-Training auf Tabellendaten |
| Friedman (2001) *Gradient Boosting* | theoretische Basis XGBoost (Ansatz 1) |
| Vabalas et al. (2019) *Validation w/ limited sample* | Begründung Nested CV |
| Crane & Crotty (1967), Malpezzi (1999) | Delta-/Residualmodell (Ansatz 2) |

---

## Ausführung im Terminal

Voraussetzung: Python ≥ 3.10. Die Rohdaten-Pfade sind in `config.py` auf deine
lokale Struktur gesetzt (kfz-repair-pricing-pipeline + Bachelorarbeit/Salesforce
Berichte). Falls Dateien woanders liegen, einfach in `config.py` anpassen.

```bash
# 1) In den Projektordner wechseln
cd kfz-repair-pricing-experiments

# 2) Virtuelle Umgebung anlegen & aktivieren
python3 -m venv .venv
source .venv/bin/activate

# 3) Abhängigkeiten installieren (scikit-learn + xgboost sind das Entscheidende)
pip install -r requirements.txt

# 4) Komplette Pipeline ausführen
python3 run_all.py
```

Einzelne Schritte (optional, z. B. zum Debuggen):

```bash
python3 src/data_prep.py        # erzeugt labeled.csv + unlabeled_pool_full.csv
python3 src/pseudo_labels.py    # erzeugt pseudo_pool_5000.csv
python3 src/descriptive.py      # Kennzahlen + Abbildungen (Kap. 7.1)
python3 src/approach1_baseline.py
python3 src/approach2_delta.py
python3 src/approach3_ssl.py
```

### Was du mir danach zurückschickst
Damit ich Kapitel 7 schreiben kann, schick mir bitte:
- den **gesamten Terminal-Output** von `python3 run_all.py`, und
- den Inhalt von `results/comparison.csv` (sowie die drei `results/approach*.csv`).

Falls ein Fehler auftritt: einfach die komplette Fehlermeldung schicken, ich passe
den Code an.
