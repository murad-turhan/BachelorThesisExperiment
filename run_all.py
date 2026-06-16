"""
Orchestrator: komplette Experiment-Pipeline (Kap. 6/7).

  1. data_prep        -> labeled.csv + unlabeled_pool_full.csv
  2. pseudo_labels    -> pseudo_pool_5000.csv (stratifiziert)
  3. descriptive      -> Kennzahlen + Abbildungen (Kap. 7.1)
  4. Ansatz 1/2/3     -> Nested-CV-Ergebnisse (je Fold + Mittel)
  5. Gesamtvergleich  -> results/comparison.csv + model_results.json + Abbildung

Aufruf (aus dem Projektordner):  python3 run_all.py
Optionale SSL-Diagnose (Kap. 7.4): python3 src/approach3_ssl_diagnostics.py
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import config as C  # noqa: E402
import pandas as pd  # noqa: E402
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def main():
    from src import data_prep, pseudo_labels, descriptive
    from src import approach1_baseline, approach2_delta, approach3_ssl

    print("\n########## 1) Datenaufbereitung ##########")
    data_prep.main()
    print("\n########## 2) Stratified Pseudo-Label-Auswahl ##########")
    pseudo_labels.main()
    print("\n########## 3) Deskriptive Analyse ##########")
    descriptive.run()

    print("\n########## 4) Modellansätze ##########")
    f1, s1 = approach1_baseline.run()
    f2, s2 = approach2_delta.run()
    f3, s3 = approach3_ssl.run()

    comp = pd.concat([s1, s2, s3], ignore_index=True)
    keep = ["approach", "model", "MAE_mean", "MAE_std", "RMSE_mean", "RMSE_std",
            "R2_mean", "R2_std", "MAPE_mean", "MAPE_std"]
    comp = comp[keep].round(3)
    comp.to_csv(C.RESULTS_DIR / "comparison.csv", index=False)

    all_folds = pd.concat([f1, f2, f3], ignore_index=True)
    (C.RESULTS_DIR / "model_results.json").write_text(json.dumps({
        "random_state": C.RANDOM_STATE,
        "outer_folds": C.OUTER_FOLDS, "inner_folds": C.INNER_FOLDS,
        "summary": comp.to_dict(orient="records"),
        "per_fold": all_folds.to_dict(orient="records"),
    }, ensure_ascii=False, indent=2))

    print("\n########## 5) Gesamtvergleich (Kap. 7.5) ##########")
    print(comp[["approach", "model", "MAE_mean", "RMSE_mean", "R2_mean", "MAPE_mean"]]
          .to_string(index=False))

    # MAE-Vergleichsabbildung (mit Std über Folds)
    main_models = comp[comp["model"].str.contains(
        "Ridge|RandomForest|XGBoost|RF_SSL", regex=True)].copy()
    # Saubere, an die Thesis-Tabellen (A1/A2/A3) angelehnte Achsenbeschriftung
    _approach_short = {"1_baseline": "A1", "2_delta": "A2", "3_ssl": "A3"}
    _model_short = {"Ridge": "Ridge", "RandomForest": "RF", "XGBoost": "XGB",
                    "RF_SSL(labels+pseudo)": "RF+SSL"}
    labels = [f"{_approach_short.get(a, a)}: {_model_short.get(m, m)}"
              for a, m in zip(main_models["approach"], main_models["model"])]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(range(len(main_models)), main_models["MAE_mean"],
           yerr=main_models["MAE_std"], capsize=4, color="#1F3864")
    ax.set_xticks(range(len(main_models)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_ylabel("MAE (Euro)"); ax.set_title("MAE-Vergleich aller Ansätze (± Std über 5 Folds)")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "mae_comparison.png", dpi=140)
    fig.savefig(C.FIGURES_DIR / "mae_comparison.pdf"); plt.close(fig)

    # Lesbarer Gesamtreport mit allen Tabellen (results/RESULTS.md)
    from src import report
    report.main()

    print(f"\nAlle Ergebnisse: {C.RESULTS_DIR}")
    print(f"Alle Abbildungen: {C.FIGURES_DIR}")


if __name__ == "__main__":
    main()
