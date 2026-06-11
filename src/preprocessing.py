"""
Vorverarbeitung (Kap. 5.2 / Listing lst:preprocessor).

Numerische Merkmale: Median-Imputation + StandardScaler.
Kategoriale Merkmale: Konstant-Imputation + OneHotEncoding.

Hinweis: Bei den hochkardinalen Merkmalen (model, motor_code) treten in den
äußeren Test-Folds zwangsläufig ungesehene Ausprägungen auf. Daher wird
handle_unknown="ignore" verwendet; unbekannte Kategorien werden als Nullvektor
kodiert. (Die im Thesis-Listing genannte Option drop="first" ist mit
handle_unknown="ignore" in scikit-learn nicht kombinierbar und entfällt deshalb.)

Der ColumnTransformer ist Teil der sklearn-Pipeline und wird in der Nested CV
ausschließlich auf den Trainings-Folds gefittet -> kein Data Leakage.
"""
from __future__ import annotations
import sys
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def make_preprocessor(numeric, categorical) -> ColumnTransformer:
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:  # ältere sklearn-Versionen
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="unbekannt")),
        ("onehot", ohe),
    ])
    return ColumnTransformer([
        ("num", num_pipe, list(numeric)),
        ("cat", cat_pipe, list(categorical)),
    ], remainder="drop")
