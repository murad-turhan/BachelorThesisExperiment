"""
Ansatz 1: Data-Driven Baseline-Regression (Kap. 5.3 / 7.2).

Direkte Regression final_price ~ Fahrzeug-/Schadensmerkmale, OHNE
Kostenvoranschlag als Feature (vgl. Kap. 5.2). Modelle: Ridge, RandomForest,
XGBoost. Bewertung über Nested CV 5x3 (Metriken je Fold + Mittel).
"""
from __future__ import annotations
import sys
import json
import pandas as pd

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.models import get_models
from src.nested_cv import run_nested_cv


def run():
    df = pd.read_csv(C.LABELED_CSV)
    num = C.numeric_features(with_kv=False)
    cat = C.categorical_features()
    X = df[num + cat].copy()
    y = df[C.TARGET].values

    per_fold, summary = run_nested_cv(X, y, get_models(), num, cat,
                                      target_is_delta=False)
    summary.insert(0, "approach", "1_baseline")
    per_fold.insert(0, "approach", "1_baseline")
    per_fold.to_csv(C.RESULTS_DIR / "approach1_folds.csv", index=False)
    summary.to_csv(C.RESULTS_DIR / "approach1_baseline.csv", index=False)
    print("\n=== Ansatz 1: Baseline-Regression (ohne KV) -- Nested CV 5x3 ===")
    print("--- Metriken je Fold ---")
    print(per_fold[["model", "fold", "MAE", "RMSE", "R2", "MAPE", "params"]]
          .round(3).to_string(index=False))
    print("--- Mittel über Folds ---")
    print(summary[["model", "MAE_mean", "RMSE_mean", "R2_mean", "MAPE_mean"]]
          .round(3).to_string(index=False))
    return per_fold, summary


if __name__ == "__main__":
    run()
