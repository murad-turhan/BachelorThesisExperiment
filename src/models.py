"""
Modelldefinitionen und Hyperparameter-Suchräume (Kap. 6.3, Tab. tab:hyperparams).

  - Ridge           : alpha in {0.01, 0.1, 1, 10, 100}
  - RandomForest    : n_estimators {100,300}, max_depth {3,5,None}, min_samples_leaf {1,3,5}
  - XGBoost         : n_estimators {100,300}, max_depth {3,5}, learning_rate {0.05,0.1,0.3},
                      subsample {0.7,1.0}

Parameternamen mit Prefix 'model__', da der Estimator in einer Pipeline
('pre' -> 'model') steckt.
"""
from __future__ import annotations
import sys
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except Exception:  # pragma: no cover
    HAS_XGB = False


def get_models() -> dict:
    """Gibt {name: (estimator, param_grid)} zurück."""
    models = {
        "Ridge": (
            Ridge(),
            {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]},
        ),
        "RandomForest": (
            RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [3, 5, None],
                "model__min_samples_leaf": [1, 3, 5],
            },
        ),
    }
    if HAS_XGB:
        models["XGBoost"] = (
            XGBRegressor(
                random_state=C.RANDOM_STATE, n_jobs=-1,
                objective="reg:squarederror", tree_method="hist",
            ),
            {
                "model__n_estimators": [100, 300],
                "model__max_depth": [3, 5],
                "model__learning_rate": [0.05, 0.1, 0.3],
                "model__subsample": [0.7, 1.0],
            },
        )
    return models
