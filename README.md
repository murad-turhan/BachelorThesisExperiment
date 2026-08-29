# Model experiments for the bachelor thesis
**Repair cost prediction at deinmotorschaden.de under small-data conditions**

This repository holds the full, reproducible code for the empirical experiments
(Chapter 6 *Implementation* and Chapter 7 *Empirical Results*). The implementation
follows the methodology described in Chapter 5.

For data protection reasons (GDPR) the repository contains only code. The
underlying customer data is not published.

## 1. Mapping to the thesis

| File | Chapter | Content |
|---|---|---|
| `src/data_prep.py` | 4.2 to 4.4 / 5.2 | Labeled set and unlabeled pool from the Salesforce exports |
| `src/pseudo_labels.py` | 5.5 | Stratified selection of the 5,000 pseudo-label candidates |
| `src/preprocessing.py` | 5.2 | StandardScaler (numeric) and OneHotEncoding (categorical) |
| `src/models.py` | 5.3 / 6.3 | Ridge, RandomForest, XGBoost plus the hyperparameter grids |
| `src/metrics.py` | 5.6 | MAE, RMSE, R², MAPE |
| `src/nested_cv.py` | 5.6 | Nested CV (outer 5-fold, inner 3-fold) |
| `src/approach1_baseline.py` | 5.3 / 7.2 | Approach 1: baseline regression |
| `src/approach2_delta.py` | 5.4 / 7.3 | Approach 2: delta / residual modelling |
| `src/approach3_ssl.py` | 5.5 / 7.4 | Approach 3: SSL with pseudo-labeling |
| `src/approach3_ssl_diagnostics.py` | 7.4 | Iteration and threshold diagnostics for the SSL approach |
| `src/descriptive.py` | 7.1 | Descriptive analysis and figures |
| `run_all.py` | 6/7 | Orchestrates the whole pipeline |

## 2. Project structure

```
kfz-repair-pricing-experiments/
├── config.py                 # central paths and parameters (seed, folds, ...)
├── requirements.txt
├── run_all.py                # end-to-end run
├── data/                     # generated datasets (local, not in the repo)
├── results/                  # nested-CV results as CSV/JSON
├── figures/                  # figures for Ch. 7.1 (local, not in the repo)
└── src/                      # modules (see the table above)
```

## 3. Data basis

Three Salesforce objects are joined over relational IDs (Ch. 4.2). One note on the
join: the offers export uses 18-character lead IDs, the customer export uses
15-character ones, so the join runs on the first 15 characters.

**Labeled set (approaches 1 and 2): 112 cases** with a final repair price confirmed
by one of the seven workshops.

**Unlabeled pool (approach 3): roughly 38,000 cost estimates.**
`offers_with_estimates.xlsx` (all cost estimates) is joined with the full
`Customer` export (vehicle features) over the lead ID; already labeled leads are
excluded.

### Feature selection (Ch. 5.2)

The twelve structured features used in the thesis are:

- **Numeric:** `year_of_construction`, `mileage`, `kw`, `hsn`, `vehicle_age`
- **Categorical:** `brand`, `model`, `fuel_type`, `car_from_country`,
  `vehicle_ready_to_drive`, `damage_type`, `motor_code`
- **Cost estimate (`price_estimation`):** only used as a feature from approach 2
  onward (deliberately left out in approach 1, see Ch. 5.2).

Numeric features get median imputation plus `StandardScaler`. Categorical features
get constant imputation plus `OneHotEncoder(handle_unknown="ignore")`. Because of
the high-cardinality fields (`model`, `motor_code`), unseen categories show up in
the test folds; `handle_unknown="ignore"` catches those. Every step is fitted
inside the fold, so there is no data leakage.

## 4. The three approaches

**Approach 1: baseline regression (Ch. 5.3).** Direct regression
`final_price ~ vehicle and damage features` (without the cost estimate) using
Ridge, RandomForest and XGBoost.

**Approach 2: delta / residual modelling (Ch. 5.4).** The target is
`delta = final_price - KV`, and the prediction is
`final_price_hat = KV + delta_hat`. Evaluation happens on the reconstructed price,
so it stays directly comparable to approach 1. Theoretical basis: Crane and Crotty
(1967), Malpezzi (1999). In practice the delta transformation roughly halves the
variance the model has to learn (Var(final) is about 19.0 million, Var(delta) about
9.2 million).

**Approach 3: SSL with pseudo-labeling (Ch. 5.5).** Self-training with a confidence
threshold (Lee 2013):

1. Tune a RandomForest on the labeled training fold (inner 3-fold).
2. Predict `final_price` on the 5,000 cost estimates; the uncertainty is the
   standard deviation across the RF trees.
3. The most certain `q %` (lowest std) are accepted as pseudo-labels.
4. Add the accepted ones to the training set and refit, for at most 3 iterations.

The threshold `q` in {20, 30, 40 %} is tuned inside the inner CV. Leakage
protection: pseudo-labels come only from the outer training fold, and the test fold
is never touched. To address the performance-degradation question (van Engelen and
Hoos 2020), a purely supervised RandomForest is also evaluated per fold.

## 5. Validation (Ch. 5.6)

Nested cross-validation: an outer `KFold(5, shuffle, seed=42)` for an unbiased
performance estimate, and an inner `GridSearchCV(cv=3)` for the hyperparameter
choice. The full preprocessing sits inside the scikit-learn pipeline and is refitted
per fold. The reasoning behind this strategy at n = 112 follows Vabalas et al.
(2019).

## 6. Source mapping

| Source | Use in the code |
|---|---|
| Lee (2013) *Pseudo-Label* | Self-training principle (approach 3) |
| Rizve et al. (2021) *Uncertainty-Aware PL Selection* | Motivation for filtering by uncertainty rather than raw confidence; UPS itself is not implemented |
| van Engelen and Hoos (2020) *SSL Survey* | Supervised reference / performance-degradation test |
| Kim et al. (2023) *Self-Training for Tabular* | SSL self-training on tabular data |
| Friedman (2001) *Gradient Boosting* | Theoretical basis for XGBoost (approach 1) |
| Vabalas et al. (2019) *Validation w/ limited sample* | Reasoning for nested CV |
| Crane and Crotty (1967), Malpezzi (1999) | Delta / residual model (approach 2) |

## 7. How to run

Requirement: Python 3.10 or newer. The paths to the raw data are configurable in
`config.py` (via an environment variable or directly in the code).

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python3 run_all.py
python3 src/approach3_ssl_diagnostics.py   # SSL diagnostics (Ch. 7.4)
```

All results are written to `results/`, all figures to `figures/`. Thanks to the
fixed `random_state=42` every run is reproducible.

Individual steps can also be run on their own, for example:

```bash
python3 src/data_prep.py        # labeled.csv + unlabeled_pool_full.csv
python3 src/pseudo_labels.py    # pseudo_pool_5000.csv
python3 src/descriptive.py      # metrics and figures
python3 src/approach1_baseline.py
python3 src/approach2_delta.py
python3 src/approach3_ssl.py
```
