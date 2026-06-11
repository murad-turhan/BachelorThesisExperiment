# Modell-Experimente zur Bachelorarbeit
**Reparaturkostenprognose bei deinmotorschaden.de unter Small-Data-Bedingungen**

Dieses Repository enthält den vollständigen, reproduzierbaren Code für die
empirischen Experimente (Kapitel 6 *Implementierung* und Kapitel 7 *Empirische
Ergebnisse*). Die Umsetzung folgt exakt der Methodik aus Kapitel 5.

Aus Datenschutzgründen (DSGVO) enthält das Repository ausschließlich Code; die
zugrundeliegenden Kundendaten werden nicht veröffentlicht.

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
| `src/approach3_ssl_diagnostics.py` | 7.4 | Iterations-/Schwellwert-Diagnose des SSL-Ansatzes |
| `src/descriptive.py` | 7.1 | Deskriptive Analyse + Abbildungen |
| `run_all.py` | 6/7 | Orchestriert die gesamte Pipeline |

---

## 2. Projektstruktur

```
kfz-repair-pricing-experiments/
├── config.py                 # zentrale Pfade & Parameter (Seed, Folds, ...)
├── requirements.txt
├── run_all.py                # End-to-End-Lauf
├── data/                     # erzeugte Datensätze (lokal, nicht im Repo)
├── results/                  # Nested-CV-Ergebnisse als CSV/JSON
├── figures/                  # Abbildungen für Kap. 7.1 (lokal, nicht im Repo)
└── src/                      # Module (s. Tabelle oben)
```

---

## 3. Datengrundlage

Drei Salesforce-Objekte werden über relationale IDs verknüpft (Kap. 4.2). Hinweis:
Der Offers-Export nutzt 18-stellige, der Customer-Export 15-stellige Lead-IDs –
gejoint wird auf den ersten 15 Zeichen.

**Labeled-Set (Ansatz 1 & 2): 112 Fälle** mit von einer der sieben Werkstätten
bestätigtem finalem Reparaturpreis.

**Unlabeled-Pool (Ansatz 3): rund 38.000 KVs.** `offers_with_estimates.xlsx`
(alle Kostenvoranschläge) wird mit dem vollständigen `Customer`-Export
(Fahrzeug-Features) über die Lead-ID verknüpft; gelabelte Leads werden
ausgeschlossen.

### Feature-Selektion (Kap. 5.2)
Verwendet werden die zwölf strukturierten Features der Arbeit:

- **Numerisch:** `year_of_construction`, `mileage`, `kw`, `hsn`, `vehicle_age`
- **Kategorial:** `brand`, `model`, `fuel_type`, `car_from_country`,
  `vehicle_ready_to_drive`, `damage_type`, `motor_code`
- **Kostenvoranschlag (`price_estimation`):** erst ab Ansatz 2 ein Feature
  (in Ansatz 1 bewusst ausgeschlossen, vgl. Kap. 5.2).

Numerische Features: Median-Imputation + `StandardScaler`. Kategoriale Features:
Konstant-Imputation + `OneHotEncoder(handle_unknown="ignore")`. Wegen der
hochkardinalen Felder (`model`, `motor_code`) treten in Test-Folds ungesehene
Ausprägungen auf; `handle_unknown="ignore"` fängt diese ab. Alle Schritte werden
fold-intern gefittet (kein Data Leakage).

---

## 4. Die drei Ansätze

**Ansatz 1 – Baseline-Regression (Kap. 5.3).** Direkte Regression
`final_price ~ Fahrzeug-/Schadensmerkmale` (ohne KV) mit Ridge / RandomForest /
XGBoost.

**Ansatz 2 – Delta-/Residual-Modellierung (Kap. 5.4).** Zielvariable ist
`delta = final_price − KV`; vorhergesagt wird `final_price_hat = KV + delta_hat`.
Die Bewertung erfolgt auf dem rekonstruierten Preis und ist damit direkt mit
Ansatz 1 vergleichbar. Theoretische Basis: Crane & Crotty (1967), Malpezzi (1999).
Empirisch halbiert die Delta-Transformation die zu lernende Varianz
(Var(final) ≈ 19,0 Mio. → Var(delta) ≈ 9,2 Mio.).

**Ansatz 3 – SSL mit Pseudo-Labeling (Kap. 5.5).** Self-Training mit
Confidence-Schwellwert (Lee 2013; Rizve et al. 2021):
1. RandomForest auf dem gelabelten Trainings-Fold tunen (innerer 3-fold).
2. Vorhersage `final_price` auf den 5.000 KVs; Unsicherheit = Std. über die RF-Bäume.
3. Die sichersten `q %` (niedrigste Std.) werden als Pseudo-Labels akzeptiert.
4. Akzeptierte ins Training übernehmen, neu fitten — max. 3 Iterationen.

Der Schwellwert `q ∈ {20, 30, 40 %}` wird im inneren CV mitoptimiert. Leakage-Schutz:
Pseudo-Labels entstehen ausschließlich aus dem äußeren Trainings-Fold; der Test-Fold
bleibt unangetastet. Zur Performance-Degradation-Frage (van Engelen & Hoos 2020)
wird je Fold zusätzlich ein rein überwachter RandomForest bewertet.

---

## 5. Validierung (Kap. 5.6)

Nested Cross-Validation: äußerer `KFold(5, shuffle, seed=42)` zur unverzerrten
Performanzschätzung, innerer `GridSearchCV(cv=3)` zur Hyperparameterwahl. Die
gesamte Vorverarbeitung steckt in der scikit-learn-Pipeline und wird pro Fold neu
gefittet. Begründung der Strategie bei n = 112: Vabalas et al. (2019).

---

## 6. Quellen-Mapping

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

## 7. Ausführung

Voraussetzung: Python ≥ 3.10. Die Pfade zu den Rohdaten sind in `config.py`
konfigurierbar (per Umgebungsvariable oder direkt im Code).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run_all.py
python3 src/approach3_ssl_diagnostics.py   # SSL-Diagnose (Kap. 7.4)
```

Alle Ergebnisse werden nach `results/`, alle Abbildungen nach `figures/`
geschrieben. Durch den festen `random_state=42` ist jeder Lauf reproduzierbar.

Einzelne Schritte lassen sich auch separat ausführen, z. B.:

```bash
python3 src/data_prep.py        # labeled.csv + unlabeled_pool_full.csv
python3 src/pseudo_labels.py    # pseudo_pool_5000.csv
python3 src/descriptive.py      # Kennzahlen + Abbildungen
python3 src/approach1_baseline.py
python3 src/approach2_delta.py
python3 src/approach3_ssl.py
```
