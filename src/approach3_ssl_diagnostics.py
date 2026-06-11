"""
SSL-Diagnose (Kap. 7.4): Entwicklung der Modellgüte
  (a) über die Self-Training-Iterationen und
  (b) über die Akzeptanzquote q (Anteil je Iteration übernommener Pseudo-Labels).

Untermauert den SSL-Befund robust. Methodik identisch zu approach3_ssl
(Nested CV, gleiche äußere Folds, Leakage-Schutz).

Ausgaben:
  results/ssl_iteration_trajectory.csv
  results/ssl_threshold_sweep.csv
  figures/ssl_iterations.png, figures/ssl_threshold_sweep.png
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold, GridSearchCV

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402
from src import metrics as M
from src.approach3_ssl import _rf_pipe, _tree_uncertainty, RF_GRID, FEATURES


def trajectory(X_lab, y_lab, X_pool, params, X_test, y_test, q, max_iter):
    X_train = X_lab.copy()
    y_train = np.asarray(y_lab, float).copy()
    pool = X_pool.reset_index(drop=True).copy()
    pipe = _rf_pipe(params).fit(X_train, y_train)
    traj = [{"iteration": 0, "n_pseudo_cum": 0,
             **M.all_metrics(y_test, pipe.predict(X_test))}]
    n_cum = 0
    for it in range(1, max_iter + 1):
        if len(pool) == 0:
            break
        mean_pred, std_pred = _tree_uncertainty(pipe, pool[FEATURES])
        thr = np.quantile(std_pred, q)
        keep = std_pred <= thr
        if keep.sum() == 0:
            break
        X_train = pd.concat([X_train, pool.loc[keep, FEATURES]], ignore_index=True)
        y_train = np.concatenate([y_train, mean_pred[keep]])
        n_cum += int(keep.sum())
        pool = pool.loc[~keep].reset_index(drop=True)
        pipe = _rf_pipe(params).fit(X_train, y_train)
        traj.append({"iteration": it, "n_pseudo_cum": n_cum,
                     **M.all_metrics(y_test, pipe.predict(X_test))})
    return traj


def run():
    lab = pd.read_csv(C.LABELED_CSV)
    pool_full = pd.read_csv(C.PSEUDO_POOL_CSV)
    X = lab[FEATURES].reset_index(drop=True)
    y = lab[C.TARGET].values.astype(float)
    outer = KFold(n_splits=C.OUTER_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)

    records = []
    for k, (tr, te) in enumerate(outer.split(X), start=1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y[tr], y[te]
        inner = KFold(n_splits=C.INNER_FOLDS, shuffle=True, random_state=C.RANDOM_STATE)
        gs = GridSearchCV(_rf_pipe(), RF_GRID, scoring="neg_mean_absolute_error",
                          cv=inner, n_jobs=-1).fit(Xtr, ytr)
        params = gs.best_params_
        for q in C.SSL_THRESHOLD_SWEEP:
            for row in trajectory(Xtr, ytr, pool_full, params, Xte, yte,
                                  q=q, max_iter=C.SSL_DIAG_MAX_ITER):
                records.append({"q": q, "fold": k, **row})

    df = pd.DataFrame(records)
    cols = ["MAE", "RMSE", "R2", "MAPE", "n_pseudo_cum"]
    traj_mean = df.groupby(["q", "iteration"])[cols].mean().round(2).reset_index()
    traj_mean.to_csv(C.RESULTS_DIR / "ssl_iteration_trajectory.csv", index=False)
    last = df.sort_values("iteration").groupby(["q", "fold"]).tail(1)
    sweep = last.groupby("q")[cols].mean().round(2).reset_index()
    sweep.to_csv(C.RESULTS_DIR / "ssl_threshold_sweep.csv", index=False)
    sup = df[df["iteration"] == 0].groupby("fold")[["MAE", "RMSE", "R2", "MAPE"]].mean().mean()

    print("=== SSL-Diagnose: Iterations-Trajektorie (über Folds gemittelt) ===")
    print(traj_mean.to_string(index=False))
    print("\n=== SSL-Diagnose: Schwellwert-Sweep (Endmetriken je q) ===")
    print(sweep.to_string(index=False))
    print(f"\nReferenz rein überwacht (Iteration 0): MAE={sup['MAE']:.2f} "
          f"RMSE={sup['RMSE']:.2f} R2={sup['R2']:.2f} MAPE={sup['MAPE']:.2f}")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for q in C.SSL_THRESHOLD_SWEEP:
        sub = traj_mean[traj_mean["q"] == q]
        ax.plot(sub["iteration"], sub["MAE"], marker="o", label=f"q = {q:.2f}")
    ax.axhline(sup["MAE"], color="grey", ls="--", lw=1, label="überwacht (Iter. 0)")
    ax.set_xlabel("Self-Training-Iteration"); ax.set_ylabel("MAE (Euro)")
    ax.set_title("Ansatz 3: MAE-Verlauf über Iterationen je Akzeptanzquote q")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "ssl_iterations.png", dpi=140); plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(sweep["q"], sweep["MAE"], "o-", color="#C62828", label="MAE")
    ax1.axhline(sup["MAE"], color="grey", ls="--", lw=1)
    ax1.set_xlabel("Akzeptanzquote q"); ax1.set_ylabel("MAE (Euro)", color="#C62828")
    ax2 = ax1.twinx(); ax2.plot(sweep["q"], sweep["R2"], "s-", color="#1F3864")
    ax2.set_ylabel("R²", color="#1F3864")
    ax1.set_title("Ansatz 3: Modellgüte über den Confidence-Schwellwert")
    fig.tight_layout(); fig.savefig(C.FIGURES_DIR / "ssl_threshold_sweep.png", dpi=140)
    plt.close(fig)
    return traj_mean, sweep


if __name__ == "__main__":
    run()
