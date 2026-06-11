"""
Auswahl der 5.000 Pseudo-Label-Kandidaten (Kap. 5.5).

Ziel: Aus dem großen KV-Pool (~38.600 ungelabelte Fälle) eine Teilmenge von
max. 5.000 ziehen, deren Kovariaten-Verteilung auf die ~112 ECHTEN Labels
abgestimmt ist ("distribution matching").

Begründung (rote Faden zu Kap. 2.4 / 5.5):
Semi-Supervised Learning stützt sich auf die Cluster-/Smoothness-Annahme – ein
Pseudo-Label-Pool, der dieselbe Verteilung wie die gelabelten Daten besitzt,
reduziert den Covariate-Shift zwischen gelabelten und pseudo-gelabelten Daten und
macht die SSL-Annahmen plausibler. Würde man rein zufällig ziehen, wäre der Pool
BMW-/Opel-lastig, während die Labels Mercedes-/VW-lastig sind – die Modelle würden
dann Pseudo-Labels in einem Feature-Bereich erzeugen, den sie kaum aus echten
Labels kennen.

Vorgehen:
  1. Brand-Anteile der gelabelten Daten bestimmen.
  2. Diese Anteile auf 5.000 hochrechnen (proportionale Allokation).
  3. Innerhalb jeder Marke zufällig (fester Seed) aus dem Pool ziehen.
  4. Rundungsrest auffüllen, Alignment verifizieren.
"""
from __future__ import annotations
import sys
import warnings
import numpy as np
import pandas as pd

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

warnings.filterwarnings("ignore")


def allocate(labeled: pd.DataFrame, pool: pd.DataFrame,
             n: int = C.N_PSEUDO, col: str = C.STRATIFY_COL,
             seed: int = C.RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.RandomState(seed)
    shares = labeled[col].value_counts(normalize=True)
    pool_counts = pool[col].value_counts()

    # nur Marken, die im Pool verfügbar sind; Anteile renormalisieren
    shares = shares[shares.index.isin(pool_counts.index)]
    shares = shares / shares.sum()

    target = (shares * n).round().astype(int)
    chosen = []
    for brand, k in target.items():
        avail = pool[pool[col] == brand]
        take = min(k, len(avail))
        if take > 0:
            chosen.append(avail.sample(n=take, random_state=rng))
    picked = pd.concat(chosen) if chosen else pool.iloc[:0]

    # Rundungs-/Engpass-Rest proportional aus dem Rest des Pools auffüllen
    remaining = n - len(picked)
    if remaining > 0:
        rest = pool.drop(index=picked.index)
        if len(rest) > 0:
            take = min(remaining, len(rest))
            picked = pd.concat([picked, rest.sample(n=take, random_state=rng)])

    return picked.sample(frac=1, random_state=rng).reset_index(drop=True)


def alignment_report(labeled: pd.DataFrame, sample: pd.DataFrame,
                     col: str = C.STRATIFY_COL) -> pd.DataFrame:
    lab = (labeled[col].value_counts(normalize=True) * 100).round(1)
    smp = (sample[col].value_counts(normalize=True) * 100).round(1)
    rep = pd.concat([lab, smp], axis=1, keys=["labeled_%", "pseudo_%"]).fillna(0.0)
    rep["diff_pp"] = (rep["pseudo_%"] - rep["labeled_%"]).round(1)
    return rep.sort_values("labeled_%", ascending=False)


def main():
    labeled = pd.read_csv(C.LABELED_CSV)
    pool = pd.read_csv(C.UNLABELED_POOL_CSV)
    sample = allocate(labeled, pool)
    sample.to_csv(C.PSEUDO_POOL_CSV, index=False)
    print(f"[pseudo_labels] gezogen: {len(sample)} Kandidaten -> {C.PSEUDO_POOL_CSV.name}")
    print("\nMarken-Alignment (labeled vs. gezogene 5.000):")
    print(alignment_report(labeled, sample).to_string())
    # Sekundär-Diagnostik: KV-Preis-Verteilung
    print("\nKV-Preis (price_estimation) – Median/Mean:")
    print(f"  labeled: median={labeled['price_estimation'].median():.0f} "
          f"mean={labeled['price_estimation'].mean():.0f}")
    print(f"  pseudo : median={sample['price_estimation'].median():.0f} "
          f"mean={sample['price_estimation'].mean():.0f}")


if __name__ == "__main__":
    main()
