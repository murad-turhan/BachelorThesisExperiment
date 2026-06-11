"""
Zentrale Konfiguration für die Modell-Experimente (Kapitel 6/7 der Bachelorarbeit).

Alle Pfade und globalen Parameter an EINER Stelle, damit die Experimente
reproduzierbar sind (vgl. Kap. 6.4 "Reproduzierbarkeit, Parametrisierung und Logging").

Pfade können per Umgebungsvariable überschrieben werden, sind aber standardmäßig
auf die lokale Ordnerstruktur von Murads Rechner gesetzt.
"""
from __future__ import annotations
import os
from pathlib import Path

# --- Projektordner -----------------------------------------------------------
PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data"
RESULTS_DIR = PROJECT_DIR / "results"
FIGURES_DIR = PROJECT_DIR / "figures"
for _d in (DATA_DIR, RESULTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Rohdatenquellen (Salesforce-Exporte) ------------------------------------
# Können per ENV überschrieben werden, falls die Dateien woanders liegen.
KFZ_REPO = Path(os.environ.get(
    "KFZ_REPO",
    "/Users/muradturhan/Documents/kfz-repair-pricing-pipeline",
))
BA_DIR = Path(os.environ.get(
    "BA_DIR",
    "/Users/muradturhan/Documents/Bachelorarbeit",
))

# Labeled-Set (Ansatz 1 & 2): finale Werkstattpreise inkl. Features
WORKSHOPS_FINAL_XLSX = Path(os.environ.get(
    "WORKSHOPS_FINAL_XLSX",
    str(KFZ_REPO / "data/raw/workshops_final_prices.xlsx"),
))
# Unlabeled-Pool (Ansatz 3): alle Kostenvoranschläge + vollständiger Kundenexport
OFFERS_XLSX = Path(os.environ.get(
    "OFFERS_XLSX",
    str(KFZ_REPO / "data/raw/offers_with_estimates.xlsx"),
))
CUSTOMER_FULL_XLSX = Path(os.environ.get(
    "CUSTOMER_FULL_XLSX",
    str(BA_DIR / "Salesforce Berichte/Customer-2026-06-09-01-22-05.xlsx"),
))

# --- Aufbereitete Datensätze (werden von data_prep.py erzeugt) ---------------
LABELED_CSV = DATA_DIR / "labeled.csv"               # ~112 gelabelte Fälle
UNLABELED_POOL_CSV = DATA_DIR / "unlabeled_pool_full.csv"   # alle KVs mit Features
PSEUDO_POOL_CSV = DATA_DIR / "pseudo_pool_5000.csv"  # gezogene 5.000 Kandidaten

# --- Feature-Schema ----------------------------------------------------------
# ROHschema (wird in data_prep.py erzeugt und gespeichert).
NUMERIC_FEATURES = [
    "year_of_construction", "mileage", "kw", "hsn",
    "price_estimation", "vehicle_age",
]
CATEGORICAL_FEATURES = [
    "brand", "model", "fuel_type", "car_from_country",
    "vehicle_ready_to_drive", "damage_type", "motor_code",
]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# MODELLIERUNGS-Feature-Set (Kap. 5.2 / Tab. tab:features, Listing lst:preprocessor).
# Es werden die zwölf strukturierten Features der Thesis verwendet. Der
# Kostenvoranschlag (price_estimation) ist in Ansatz 1 KEIN Eingabemerkmal,
# sondern kommt erst ab Ansatz 2 hinzu (vgl. Kap. 5.2).
BASE_NUMERIC = ["year_of_construction", "mileage", "kw", "hsn", "vehicle_age"]
BASE_CATEGORICAL = [
    "brand", "model", "fuel_type", "car_from_country",
    "vehicle_ready_to_drive", "damage_type", "motor_code",
]


def numeric_features(with_kv: bool = False):
    """Numerische Features; ab Ansatz 2/3 inkl. Kostenvoranschlag."""
    return BASE_NUMERIC + (["price_estimation"] if with_kv else [])


def categorical_features():
    return list(BASE_CATEGORICAL)


def model_features(with_kv: bool = False):
    return numeric_features(with_kv) + categorical_features()


# Alle Spalten, die in den Datensätzen vorhanden sein müssen.
ALL_MODEL_COLUMNS = BASE_NUMERIC + ["price_estimation"] + BASE_CATEGORICAL

TARGET = "final_price"
KV_COLUMN = "price_estimation"   # Kostenvoranschlag (Engine Repair Price)
DELTA_TARGET = "delta"           # Ansatz 2: final_price - price_estimation

# --- Globale Experiment-Parameter -------------------------------------------
RANDOM_STATE = 42
OUTER_FOLDS = 5          # äußerer Loop (Performanzschätzung)
INNER_FOLDS = 3          # innerer Loop (Hyperparameterwahl)
REFERENCE_YEAR = 2025    # für vehicle_age = REFERENCE_YEAR - year_of_construction

# --- Ansatz 3: Pseudo-Labeling ----------------------------------------------
N_PSEUDO = 5000          # max. Anzahl ungelabelter KVs (Kap. 5.5)
STRATIFY_COL = "brand"   # Verteilungs-Abgleich zur Labeled-Verteilung
# Self-Training: Konfidenzmaß = Std. der Einzelbaum-Vorhersagen (RF).
# Der Schwellwert q (Anteil akzeptierter, sicherster Pseudo-Labels je Iteration)
# wird im INNEREN CV mitoptimiert; das Schema läuft über max. 3 Iterationen.
SSL_Q_GRID = [0.20, 0.30, 0.40]
SSL_MAX_ITERATIONS = 3
# Erweiterte Diagnose (Kap. 7.4): Verlauf über Iterationen und Schwellwerte.
SSL_THRESHOLD_SWEEP = [0.20, 0.30, 0.40]
SSL_DIAG_MAX_ITER = 5
