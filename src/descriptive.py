"""
Deskriptive Analyse der Datenbasis (Kap. 7.1).

Kennzahlen zu Finalpreis, KV und Delta; Marken- und Schadensartverteilung;
Verteilungsabgleich gelabelt vs. Pseudo-Pool. Nur pandas/matplotlib.
"""
from __future__ import annotations
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402


def _stats(s):
    return {"mean": s.mean(), "std": s.std(), "median": s.median(),
            "min": s.min(), "max": s.max()}


def run():
    lab = pd.read_csv(C.LABELED_CSV)
    pool = pd.read_csv(C.PSEUDO_POOL_CSV)
    fp, kv = lab[C.TARGET], lab[C.KV_COLUMN]
    delta = fp - kv

    table = pd.DataFrame({"Finalpreis": _stats(fp), "Kostenvoranschlag": _stats(kv),
                          "Delta": _stats(delta)}).T.round(2)
    table.to_csv(C.RESULTS_DIR / "descriptive_table.csv")

    brand_lab = (lab["brand"].value_counts(normalize=True) * 100).round(1)
    brand_pool = (pool["brand"].value_counts(normalize=True) * 100).round(1)
    brand_cmp = pd.concat([brand_lab, brand_pool], axis=1,
                          keys=["labeled_%", "pool_%"]).fillna(0.0)
    brand_cmp["abs_diff_pp"] = (brand_cmp["labeled_%"] - brand_cmp["pool_%"]).abs().round(1)
    max_dev = brand_cmp["abs_diff_pp"].max()

    summary = {
        "n_labeled": int(len(lab)), "n_pool": int(len(pool)),
        "corr_kv_final": float(np.corrcoef(kv, fp)[0, 1]),
        "var_final_price": float(np.var(fp)), "var_delta": float(np.var(delta)),
        "max_brand_deviation_pp": float(max_dev),
        "top_brands": lab["brand"].value_counts().head(3).to_dict(),
        "top_damage": lab["damage_type"].value_counts().head(3).to_dict(),
    }
    (C.RESULTS_DIR / "descriptive_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2))

    print("=== Deskriptive Statistiken (Kap. 7.1) ===")
    print(table.to_string())
    print(f"\nn_labeled={len(lab)}  n_pool={len(pool)}  "
          f"corr(KV,final)={summary['corr_kv_final']:.2f}")
    print(f"Var(final)={summary['var_final_price']:.0f}  Var(delta)={summary['var_delta']:.0f}")
    print(f"max. Markenabweichung labeled vs pool: {max_dev:.1f} pp")
    print("Top-Marken:", summary["top_brands"])
    print("Top-Schadensarten:", summary["top_damage"])

    # Abbildungen
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(fp, bins=20, color="#2E7D32", alpha=0.8)
    ax.axvline(fp.mean(), color="orange", ls="--", label=f"Mittelwert ({fp.mean():.0f} €)")
    ax.axvline(fp.median(), color="green", lw=2, label=f"Median ({fp.median():.0f} €)")
    ax.set_title("Verteilung der Finalpreise (n = %d)" % len(lab))
    ax.set_xlabel("Euro"); ax.legend(); fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "price_distribution.png", dpi=140)
    fig.savefig(C.FIGURES_DIR / "price_distribution.pdf"); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    b = brand_cmp.sort_values("labeled_%", ascending=False).head(8)
    b[["labeled_%", "pool_%"]].plot.bar(ax=axes[0], color=["#1F3864", "#9DB8E0"])
    axes[0].set_title("Markenverteilung: gelabelt vs. Pool"); axes[0].set_ylabel("%")
    dmg = (lab["damage_type"].value_counts(normalize=True) * 100).head(6)
    dmg.plot.bar(ax=axes[1], color="#C62828")
    axes[1].set_title("Häufigste Schadensarten (gelabelt)"); axes[1].set_ylabel("%")
    fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "distribution_check.png", dpi=140)
    fig.savefig(C.FIGURES_DIR / "distribution_check.pdf"); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(kv, fp, alpha=0.6, color="#1F3864")
    lim = [0, max(fp.max(), kv.max()) * 1.05]
    ax.plot(lim, lim, "k--", lw=1, label="KV = Finalpreis")
    ax.set_xlabel("Kostenvoranschlag (€)"); ax.set_ylabel("Finaler Preis (€)")
    ax.set_title(f"KV vs. Finalpreis (r = {summary['corr_kv_final']:.2f})")
    ax.legend(); fig.tight_layout()
    fig.savefig(C.FIGURES_DIR / "kv_vs_final.png", dpi=140); plt.close(fig)
    print(f"\nAbbildungen gespeichert unter: {C.FIGURES_DIR}")
    return table, summary


if __name__ == "__main__":
    run()
