"""
Nested Cross-Validation (Kap. 5.6).

Äußerer 5-fold-Loop: unverzerrte Performanzschätzung.
Innerer 3-fold-Loop (GridSearchCV): Hyperparameterwahl je Modelltyp.

Liefert sowohl die Metriken JE äußerem Fold (für die Tabellen in Kap. 7) als
auch die über die Folds gemittelten Kennzahlen inkl. Standardabweichung.
Die gesamte Vorverarbeitung steckt in der Pipeline und wird pro Fold neu
gefittet -> Schutz vor Optimism-Bias / Data Leakage.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.preprocessing import make_preprocessor
from src import metrics as M


def _short_params(name: str, params: dict) -> str:
    """Kompakte Darstellung der je Fold gewählten Hyperparameter (für ch7)."""
    if name == "Ridge":
        return f"alpha={params.get('model__alpha')}"
    if name == "RandomForest":
        return (f"n={params.get('model__n_estimators')},"
                f"depth={params.get('model__max_depth')},"
                f"leaf={params.get('model__min_samples_leaf')}")
    if name == "XGBoost":
        return (f"n={params.get('model__n_estimators')},"
                f"depth={params.get('model__max_depth')},"
                f"lr={params.get('model__learning_rate')},"
                f"sub={params.get('model__subsample')}")
    return str(params)


def run_nested_cv(X: pd.DataFrame, y: np.ndarray, models: dict,
                  numeric, categorical,
                  kv: np.ndarray | None = None,
                  target_is_delta: bool = False,
                  scoring: str = "neg_mean_absolute_error"):
    """Nested CV für Ansatz 1 (Baseline) und Ansatz 2 (Delta).

    Returns
    -------
    per_fold : DataFrame (model, fold, MAE, RMSE, R2, MAPE, params)
    summary  : DataFrame (model, *_mean, *_std)
    """
    outer = KFold(n_splits=C.OUTER_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    X = X.reset_index(drop=True)
    y = np.asarray(y, dtype=float)
    kv = np.asarray(kv, dtype=float) if kv is not None else None

    fold_rows, summary_rows = [], []
    for name, (estimator, grid) in models.items():
        fm = []
        for k, (tr, te) in enumerate(outer.split(X), start=1):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, yte = y[tr], y[te]
            pipe = Pipeline([("pre", make_preprocessor(numeric, categorical)),
                             ("model", estimator)])
            inner = KFold(n_splits=C.INNER_FOLDS, shuffle=True,
                          random_state=C.RANDOM_STATE)
            gs = GridSearchCV(pipe, grid, scoring=scoring, cv=inner, n_jobs=-1)
            gs.fit(Xtr, ytr)
            pred = gs.best_estimator_.predict(Xte)
            if target_is_delta:
                met = M.all_metrics(kv[te] + yte, kv[te] + pred)
            else:
                met = M.all_metrics(yte, pred)
            row = {"model": name, "fold": k, **met,
                   "params": _short_params(name, gs.best_params_)}
            fold_rows.append(row)
            fm.append(met)
        d = pd.DataFrame(fm)
        summary_rows.append({
            "model": name,
            "MAE_mean": d.MAE.mean(), "MAE_std": d.MAE.std(),
            "RMSE_mean": d.RMSE.mean(), "RMSE_std": d.RMSE.std(),
            "R2_mean": d.R2.mean(), "R2_std": d.R2.std(),
            "MAPE_mean": d.MAPE.mean(), "MAPE_std": d.MAPE.std(),
        })
    return pd.DataFrame(fold_rows), pd.DataFrame(summary_rows)
