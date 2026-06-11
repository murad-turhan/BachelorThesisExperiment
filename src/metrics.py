"""
Gütemetriken (Kap. 5.6).

Vier Metriken, jede mit eigener Aussage:
  - MAE  : mittlerer absoluter Fehler in Euro (robust, leicht interpretierbar)
  - RMSE : bestraft große Ausreißer stärker (relevant fürs Festpreisrisiko)
  - R^2  : Anteil erklärter Varianz (Vergleich gegen Mittelwert-Baseline)
  - MAPE : mittlerer prozentualer Fehler (skalenunabhängig, vergleichbar über Preisklassen)
"""
from __future__ import annotations
import numpy as np


def _arr(y):
    return np.asarray(y, dtype=float).ravel()


def mae(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def mape(y_true, y_pred) -> float:
    yt, yp = _arr(y_true), _arr(y_pred)
    mask = yt != 0
    return float(np.mean(np.abs((yt[mask] - yp[mask]) / yt[mask])) * 100)


def all_metrics(y_true, y_pred) -> dict:
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2(y_true, y_pred),
        "MAPE": mape(y_true, y_pred),
    }
