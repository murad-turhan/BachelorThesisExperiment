"""
Ansatz 2: Delta-/Residual-Modellierung (Kap. 5.4 / 7.3).

Zielgröße delta = final_price - KV; der KV fließt zusätzlich als Feature ein.
Vorhersage des Finalpreises: final_price_hat = KV + delta_hat. Bewertet wird auf
dem rekonstruierten Preis (direkt vergleichbar mit Ansatz 1).
Theoriebasis: Crane & Crotty (1967); Reduktion der Zielvariablenvarianz
(vgl. Jia et al. 2024).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.models import get_models
from src.nested_cv import run_nested_cv


def run():
    df = pd.read_csv(C.LABELED_CSV)
    num = C.numeric_features(with_kv=True)   # KV als Feature ab Ansatz 2
    cat = C.categorical_features()
    X = df[num + cat].copy()
    kv = df[C.KV_COLUMN].values.astype(float)
    delta = df[C.TARGET].values.astype(float) - kv

    print("Std final_price :", round(np.std(df[C.TARGET].values), 1))
    print("Std delta       :", round(np.std(delta), 1))

    per_fold, summary = run_nested_cv(X, delta, get_models(), num, cat,
                                      kv=kv, target_is_delta=True)
    summary.insert(0, "approach", "2_delta")
    per_fold.insert(0, "approach", "2_delta")
    per_fold.to_csv(C.RESULTS_DIR / "approach2_folds.csv", index=False)
    summary.to_csv(C.RESULTS_DIR / "approach2_delta.csv", index=False)
    print("\n=== Ansatz 2: Delta-/Residual-Modellierung -- Nested CV 5x3 ===")
    print("--- Metriken je Fold (auf rekonstruiertem Finalpreis) ---")
    print(per_fold[["model", "fold", "MAE", "RMSE", "R2", "MAPE", "params"]]
          .round(3).to_string(index=False))
    print("--- Mittel über Folds ---")
    print(summary[["model", "MAE_mean", "RMSE_mean", "R2_mean", "MAPE_mean"]]
          .round(3).to_string(index=False))
    return per_fold, summary


if __name__ == "__main__":
    run()
