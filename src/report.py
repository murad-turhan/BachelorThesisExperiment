"""
Report-Generator: fasst alle Ergebnis-CSVs zu einer lesbaren Markdown-Datei
`results/RESULTS.md` zusammen (rendert auf GitHub automatisch als Tabellen).

Liest die von run_all.py erzeugten CSVs und schreibt:
  - deskriptive Statistiken (Kap. 7.1)
  - Ø-Metriken + Fold-Tabellen je Ansatz (Kap. 7.2–7.4)
  - SSL-Schwellwert-Diagnose (Kap. 7.4)
  - Gesamtvergleich (Kap. 7.5)
Benötigt nur pandas.
"""
from __future__ import annotations
import sys
import json
import pandas as pd

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def df_to_md(df: pd.DataFrame, floatfmt="{:.3f}") -> str:
    """DataFrame -> Markdown-Tabelle (ohne externe Abhängigkeiten)."""
    cols = list(df.columns)
    def fmt(v):
        if isinstance(v, float):
            return floatfmt.format(v)
        return str(v)
    head = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |"
            for r in df.itertuples(index=False)]
    return "\n".join([head, sep] + rows)


def _read(name):
    p = C.RESULTS_DIR / name
    return pd.read_csv(p) if p.exists() else None


def main():
    out = ["# Ergebnisse der Experimente",
           "",
           "Nested CV 5×3, n = 112 gelabelte Fälle, Seed 42. Automatisch erzeugt "
           "aus den Ergebnis-CSVs (`src/report.py`).", ""]

    desc = _read("descriptive_table.csv")
    if desc is not None:
        desc = desc.rename(columns={desc.columns[0]: "Variable"})
        out += ["## 1. Datengrundlage (Kap. 7.1)", "",
                df_to_md(desc, "{:.2f}"), ""]
        sj = C.RESULTS_DIR / "descriptive_summary.json"
        if sj.exists():
            s = json.loads(sj.read_text())
            out += ["",
                    f"- n (gelabelt) = {s.get('n_labeled')}, Pool = {s.get('n_pool')}",
                    f"- Korrelation KV ↔ Finalpreis: r = {s.get('corr_kv_final'):.2f}",
                    f"- Varianz Finalpreis = {s.get('var_final_price'):.0f} → "
                    f"Varianz Delta = {s.get('var_delta'):.0f}",
                    f"- max. Markenabweichung labeled vs. Pool = "
                    f"{s.get('max_brand_deviation_pp')} pp", ""]

    def approach_block(title, summary_csv, folds_csv, extra_cols=None):
        block = [f"## {title}", ""]
        s = _read(summary_csv)
        if s is not None:
            keep = [c for c in ["model", "MAE_mean", "RMSE_mean", "R2_mean", "MAPE_mean"]
                    if c in s.columns]
            block += ["**Ø-Metriken über 5 Folds**", "",
                      df_to_md(s[keep].round(3)), ""]
        f = _read(folds_csv)
        if f is not None:
            cols = [c for c in (["model", "fold", "MAE", "RMSE", "R2", "MAPE"]
                                + (extra_cols or [])) if c in f.columns]
            block += ["**Je Fold**", "", df_to_md(f[cols].round(3)), ""]
        return block

    out += approach_block("2. Ansatz 1 – Baseline (Kap. 7.2)",
                          "approach1_baseline.csv", "approach1_folds.csv")
    out += approach_block("3. Ansatz 2 – Delta-Modell (Kap. 7.3)",
                          "approach2_delta.csv", "approach2_folds.csv")
    out += approach_block("4. Ansatz 3 – SSL (Kap. 7.4)",
                          "approach3_ssl.csv", "approach3_folds.csv",
                          extra_cols=["opt_q", "n_pseudo"])

    sweep = _read("ssl_threshold_sweep.csv")
    if sweep is not None:
        out += ["**SSL-Diagnose: Schwellwert-Sweep**", "",
                df_to_md(sweep.round(2)), ""]

    comp = _read("comparison.csv")
    if comp is not None:
        keep = [c for c in ["approach", "model", "MAE_mean", "RMSE_mean",
                            "R2_mean", "MAPE_mean"] if c in comp.columns]
        out += ["## 5. Gesamtvergleich (Kap. 7.5)", "",
                df_to_md(comp[keep].round(3)), ""]

    (C.RESULTS_DIR / "RESULTS.md").write_text("\n".join(out), encoding="utf-8")
    print(f"[report] results/RESULTS.md geschrieben ({len(out)} Zeilen).")


if __name__ == "__main__":
    main()
