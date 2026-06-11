"""
Ansatz 3: Semi-Supervised Learning mit Pseudo-Labeling (Kap. 5.5 / 7.4).

Self-Training mit Konfidenz-Schwellwert (Lee 2013; Rizve et al. 2021):
  1. RandomForest auf dem aktuellen gelabelten Set (Features inkl. KV).
  2. Vorhersage final_price für den ungelabelten Pool (stratifizierte 5.000 KVs).
  3. Konfidenz = Std. der Einzelbaum-Vorhersagen (Gl. eq:konfidenz).
  4. Die sichersten q % (niedrigste Std.) werden als Pseudo-Labels übernommen.
  5. Wiederholung über max. 3 Iterationen.

Der Schwellwert q in {0.20, 0.30, 0.40} wird im INNEREN CV mitoptimiert; die
RF-Hyperparameter ebenfalls. Leakage-Schutz: Pseudo-Labels entstehen nur aus
den Trainingsdaten des jeweiligen äußeren Folds; der Test-Fold bleibt unberührt.

Zur Beurteilung der Performance-Degradation (van Engelen & Hoos 2020) wird je
Fold zusätzlich ein rein überwachter RandomForest (gleiche Hyperparameter)
bewertet.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src.preprocessing import make_preprocessor
from src import metrics as M

NUM = C.numeric_features(with_kv=True)
CAT = C.categorical_features()
FEATURES = NUM + CAT
RF_GRID = {
    "model__n_estimators": [100, 300],
    "model__max_depth": [3, 5, None],
    "model__min_samples_leaf": [1, 3, 5],
}


def _rf_pipe(params: dict | None = None) -> Pipeline:
    pipe = Pipeline([
        ("pre", make_preprocessor(NUM, CAT)),
        ("model", RandomForestRegressor(random_state=C.RANDOM_STATE, n_jobs=-1)),
    ])
    if params:
        pipe.set_params(**params)
    return pipe


def _tree_uncertainty(pipe: Pipeline, X: pd.DataFrame):
    pre = pipe.named_steps["pre"]
    rf = pipe.named_steps["model"]
    Xt = pre.transform(X)
    preds = np.stack([est.predict(Xt) for est in rf.estimators_])
    return preds.mean(axis=0), preds.std(axis=0)


def self_train(X_lab, y_lab, X_pool, params, q, max_iter=C.SSL_MAX_ITERATIONS):
    """Self-Training-Schleife. Gibt finales Modell + Anzahl Pseudo-Labels."""
    X_train = X_lab.copy()
    y_train = np.asarray(y_lab, float).copy()
    pool = X_pool.reset_index(drop=True).copy()
    n_acc = 0
    for _ in range(max_iter):
        if len(pool) == 0:
            break
        pipe = _rf_pipe(params).fit(X_train, y_train)
        mean_pred, std_pred = _tree_uncertainty(pipe, pool[FEATURES])
        thr = np.quantile(std_pred, q)
        keep = std_pred <= thr
        if keep.sum() == 0:
            break
        X_train = pd.concat([X_train, pool.loc[keep, FEATURES]], ignore_index=True)
        y_train = np.concatenate([y_train, mean_pred[keep]])
        n_acc += int(keep.sum())
        pool = pool.loc[~keep].reset_index(drop=True)
    final = _rf_pipe(params).fit(X_train, y_train)
    return final, n_acc


def _choose_q(X_tr, y_tr, pool, params):
    """Wählt q über innere 3-fold-CV (Minimierung des MAE)."""
    inner = KFold(n_splits=C.INNER_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
    best_q, best_mae = C.SSL_Q_GRID[0], np.inf
    for q in C.SSL_Q_GRID:
        maes = []
        for itr, iva in inner.split(X_tr):
            m, _ = self_train(X_tr.iloc[itr], y_tr[itr], pool, params, q)
            maes.append(M.mae(y_tr[iva], m.predict(X_tr.iloc[iva])))
        mean_mae = float(np.mean(maes))
        if mean_mae < best_mae:
            best_mae, best_q = mean_mae, q
    return best_q


def run():
    lab = pd.read_csv(C.LABELED_CSV)
    pool_full = pd.read_csv(C.PSEUDO_POOL_CSV)
    X = lab[FEATURES].reset_index(drop=True)
    y = lab[C.TARGET].values.astype(float)
    outer = KFold(n_splits=C.OUTER_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    ssl_fold, sup_fold = [], []
    for k, (tr, te) in enumerate(outer.split(X), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]

        # RF-Hyperparameter via innerer CV (auf gelabelten Trainingsdaten)
        inner = KFold(n_splits=C.INNER_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
        gs = GridSearchCV(_rf_pipe(), RF_GRID, scoring="neg_mean_absolute_error",
                          cv=inner, n_jobs=-1).fit(Xtr, ytr)
        params = gs.best_params_

        # q via innerer CV mitoptimieren
        q = _choose_q(Xtr, ytr.copy(), pool_full, params)

        # (a) SSL-Modell
        ssl_model, n_acc = self_train(Xtr, ytr, pool_full, params, q)
        m_ssl = M.all_metrics(yte, ssl_model.predict(Xte))
        m_ssl.update({"model": "RF_SSL", "fold": k, "opt_q": int(q * 100),
                      "n_pseudo": n_acc})
        ssl_fold.append(m_ssl)

        # (b) überwachte RF-Referenz
        sup = _rf_pipe(params).fit(Xtr, ytr)
        m_sup = M.all_metrics(yte, sup.predict(Xte))
        m_sup.update({"model": "RF_supervised", "fold": k})
        sup_fold.append(m_sup)

    per_fold = pd.DataFrame(ssl_fold + sup_fold)
    per_fold.insert(0, "approach", "3_ssl")
    per_fold.to_csv(C.RESULTS_DIR / "approach3_folds.csv", index=False)

    def agg(rows, name):
        d = pd.DataFrame(rows)
        return {"approach": "3_ssl", "model": name,
                "MAE_mean": d.MAE.mean(), "MAE_std": d.MAE.std(),
                "RMSE_mean": d.RMSE.mean(), "RMSE_std": d.RMSE.std(),
                "R2_mean": d.R2.mean(), "R2_std": d.R2.std(),
                "MAPE_mean": d.MAPE.mean(), "MAPE_std": d.MAPE.std()}
    summary = pd.DataFrame([agg(ssl_fold, "RF_SSL(labels+pseudo)"),
                            agg(sup_fold, "RF_supervised(only_labels)")])
    summary.to_csv(C.RESULTS_DIR / "approach3_ssl.csv", index=False)

    print("\n=== Ansatz 3: SSL mit Pseudo-Labeling -- Nested CV 5x3 ===")
    print("--- SSL je Fold (opt_q in %, n_pseudo akzeptiert) ---")
    print(pd.DataFrame(ssl_fold)[["fold", "MAE", "RMSE", "R2", "MAPE",
                                  "opt_q", "n_pseudo"]].round(3).to_string(index=False))
    print("--- Mittel über Folds ---")
    print(summary[["model", "MAE_mean", "RMSE_mean", "R2_mean", "MAPE_mean"]]
          .round(3).to_string(index=False))
    return per_fold, summary


if __name__ == "__main__":
    run()
