"""
Feature-Importance-Analyse für TF1 (Kap. 8.1 Bachelorarbeit).

Methode: Permutation Importance auf dem besten Modell (Delta-Ridge, Ansatz 2).
  - Für jeden der 5 äußeren Folds: bestes Ridge-Modell aus dem inneren CV,
    dann je Feature die Spalte im Test-Set permutieren und MAE-Anstieg messen.
  - Wiederholung: 30 Permutationen je Feature → stabiler Schätzer.
  - Aggregation der One-Hot-Kategorien zurück auf das Originalfeature
    (z.B. alle brand_* → "brand").
  - Ausgabe: CSV + Barplot (figures/feature_importance.pdf/.png).
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C
from src.preprocessing import make_preprocessor

N_REPEATS = 30
RIDGE_GRID = {"model__alpha": [0.01, 0.1, 1.0, 10.0, 100.0]}


def _permutation_importance_original(pipeline, X_test, y_test, kv_test,
                                      numeric_cols, cat_cols, n_repeats, rng):
    """
    Permutiert jedes Original-Feature (vor Preprocessing) einzeln,
    misst MAE-Anstieg. Kategorien werden danach nach Originalname aggregiert.
    """
    # Baseline MAE
    pred_base = pipeline.predict(X_test)
    mae_base = np.mean(np.abs((kv_test + y_test) - (kv_test + pred_base)))

    all_features = numeric_cols + cat_cols
    records = []
    for feat in all_features:
        maes = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[feat] = rng.permutation(X_perm[feat].values)
            pred_perm = pipeline.predict(X_perm)
            mae_perm = np.mean(np.abs((kv_test + y_test) - (kv_test + pred_perm)))
            maes.append(mae_perm - mae_base)
        records.append({"feature": feat, "importance_mean": np.mean(maes),
                        "importance_std": np.std(maes)})
    return pd.DataFrame(records)


def run():
    df = pd.read_csv(C.LABELED_CSV)
    num = C.numeric_features(with_kv=True)
    cat = C.categorical_features()
    X   = df[num + cat].reset_index(drop=True)
    kv  = df[C.KV_COLUMN].values.astype(float)
    y   = df[C.TARGET].values.astype(float) - kv   # delta

    outer = KFold(n_splits=C.OUTER_FOLDS, shuffle=True,
                  random_state=C.RANDOM_STATE)
    rng = np.random.default_rng(C.RANDOM_STATE)

    fold_imps = []
    for k, (tr, te) in enumerate(outer.split(X), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr      = y[tr]
        kv_te    = kv[te]

        pipe = Pipeline([("pre", make_preprocessor(num, cat)),
                         ("model", Ridge())])
        inner = KFold(n_splits=C.INNER_FOLDS, shuffle=True,
                      random_state=C.RANDOM_STATE)
        gs = GridSearchCV(pipe, RIDGE_GRID,
                          scoring="neg_mean_absolute_error",
                          cv=inner, n_jobs=-1)
        gs.fit(Xtr, ytr)
        best = gs.best_estimator_

        imp = _permutation_importance_original(
            best, Xte, y[te], kv_te, num, cat, N_REPEATS, rng)
        imp["fold"] = k
        fold_imps.append(imp)
        print(f"  Fold {k} fertig  (alpha={gs.best_params_['model__alpha']})")

    df_imp = pd.concat(fold_imps, ignore_index=True)
    agg = (df_imp.groupby("feature")["importance_mean"]
           .agg(["mean", "std"])
           .rename(columns={"mean": "importance_mean", "std": "importance_std"})
           .reset_index()
           .sort_values("importance_mean", ascending=False))

    agg.to_csv(C.RESULTS_DIR / "feature_importance.csv", index=False)
    print("\n=== Permutation Importance (Delta-Ridge, Ansatz 2) ===")
    print(agg.round(2).to_string(index=False))

    # ── Barplot ──────────────────────────────────────────────
    DARK  = "#1F3864"
    RED   = "#C62828"

    labels = {
        "price_estimation":       "Kostenvoranschlag (KV)",
        "year_of_construction":   "Baujahr",
        "mileage":                "Kilometerstand",
        "kw":                     "Motorleistung (kW)",
        "hsn":                    "HSN",
        "vehicle_age":            "Fahrzeugalter",
        "brand":                  "Marke",
        "model":                  "Modell",
        "damage_type":            "Schadensart",
        "fuel_type":              "Kraftstoffart",
        "motor_code":             "Motorcode",
        "car_from_country":       "Herkunftsland",
        "vehicle_ready_to_drive": "Fahrbereitschaft",
    }

    agg["label"] = agg["feature"].map(labels).fillna(agg["feature"])
    colors = [RED if r["feature"] == "price_estimation" else DARK
              for _, r in agg.iterrows()]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.barh(agg["label"], agg["importance_mean"],
                   xerr=agg["importance_std"],
                   color=colors, edgecolor="white", capsize=3,
                   error_kw={"elinewidth": 1.2, "ecolor": "#546E7A"})
    ax.axvline(0, color="black", lw=0.8)
    ax.set_xlabel("Ø MAE-Anstieg bei Permutation (€)")
    ax.set_title("Permutation Importance – Delta-Ridge (Ansatz 2)\n"
                 "gemittelt über 5 äußere Folds × 30 Permutationen")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "feature_importance.pdf", dpi=150)
    fig.savefig(C.FIGURES_DIR / "feature_importance.png", dpi=150)
    plt.close()
    print(f"\nAbbildung gespeichert: {C.FIGURES_DIR}/feature_importance.pdf")
    return agg


if __name__ == "__main__":
    run()
