# Ergebnisse der Experimente

Nested CV 5×3, n = 112 gelabelte Fälle, Seed 42. Automatisch erzeugt aus den Ergebnis-CSVs (`src/report.py`).

## 1. Datengrundlage (Kap. 7.1)

| Variable | mean | std | median | min | max |
| --- | --- | --- | --- | --- | --- |
| Finalpreis | 9887.26 | 4379.74 | 9185.61 | 2990.00 | 30589.00 |
| Kostenvoranschlag | 6071.86 | 1944.83 | 5994.50 | 1700.00 | 14999.00 |
| Delta | 3815.41 | 3050.38 | 3346.65 | -2149.00 | 17589.00 |


- n (gelabelt) = 112, Pool = 5000
- Korrelation KV ↔ Finalpreis: r = 0.80
- Varianz Finalpreis = 19010894 → Varianz Delta = 9221730
- max. Markenabweichung labeled vs. Pool = 0.7 pp

## 2. Ansatz 1 – Baseline (Kap. 7.2)

**Ø-Metriken über 5 Folds**

| model | MAE_mean | RMSE_mean | R2_mean | MAPE_mean |
| --- | --- | --- | --- | --- |
| Ridge | 2619.008 | 3772.815 | 0.160 | 30.495 |
| RandomForest | 2693.224 | 3682.640 | 0.234 | 30.929 |
| XGBoost | 2849.759 | 4032.107 | 0.065 | 31.538 |

**Je Fold**

| model | fold | MAE | RMSE | R2 | MAPE |
| --- | --- | --- | --- | --- | --- |
| Ridge | 1 | 2992.184 | 4258.619 | 0.170 | 32.047 |
| Ridge | 2 | 2431.245 | 4035.525 | 0.394 | 24.965 |
| Ridge | 3 | 2651.765 | 3861.742 | -0.163 | 27.828 |
| Ridge | 4 | 2599.761 | 3652.004 | 0.354 | 25.209 |
| Ridge | 5 | 2420.085 | 3056.183 | 0.044 | 42.426 |
| RandomForest | 1 | 2908.867 | 4139.654 | 0.216 | 30.229 |
| RandomForest | 2 | 2760.162 | 4257.047 | 0.325 | 28.386 |
| RandomForest | 3 | 2689.397 | 3315.905 | 0.142 | 28.331 |
| RandomForest | 4 | 2955.494 | 4043.838 | 0.208 | 28.061 |
| RandomForest | 5 | 2152.199 | 2656.755 | 0.278 | 39.639 |
| XGBoost | 1 | 2884.326 | 4406.833 | 0.111 | 30.518 |
| XGBoost | 2 | 3206.787 | 4959.910 | 0.084 | 30.144 |
| XGBoost | 3 | 3149.336 | 4347.506 | -0.474 | 32.112 |
| XGBoost | 4 | 2996.143 | 3909.169 | 0.260 | 28.201 |
| XGBoost | 5 | 2012.204 | 2537.115 | 0.341 | 36.714 |

## 3. Ansatz 2 – Delta-Modell (Kap. 7.3)

**Ø-Metriken über 5 Folds**

| model | MAE_mean | RMSE_mean | R2_mean | MAPE_mean |
| --- | --- | --- | --- | --- |
| Ridge | 2115.568 | 2870.789 | 0.513 | 23.523 |
| RandomForest | 2152.438 | 2884.501 | 0.510 | 24.414 |
| XGBoost | 2449.034 | 3375.646 | 0.335 | 26.653 |

**Je Fold**

| model | fold | MAE | RMSE | R2 | MAPE |
| --- | --- | --- | --- | --- | --- |
| Ridge | 1 | 2127.466 | 3062.232 | 0.571 | 23.512 |
| Ridge | 2 | 1976.853 | 3110.799 | 0.640 | 20.262 |
| Ridge | 3 | 2044.156 | 2570.243 | 0.485 | 22.280 |
| Ridge | 4 | 2350.531 | 3007.650 | 0.562 | 21.267 |
| Ridge | 5 | 2078.836 | 2603.023 | 0.307 | 30.296 |
| RandomForest | 1 | 2192.857 | 3204.008 | 0.530 | 25.219 |
| RandomForest | 2 | 1932.668 | 3086.943 | 0.645 | 19.580 |
| RandomForest | 3 | 2005.807 | 2300.853 | 0.587 | 22.357 |
| RandomForest | 4 | 2576.950 | 3170.595 | 0.513 | 23.633 |
| RandomForest | 5 | 2053.906 | 2660.105 | 0.276 | 31.282 |
| XGBoost | 1 | 2637.156 | 3746.956 | 0.358 | 28.400 |
| XGBoost | 2 | 2184.214 | 3653.818 | 0.503 | 20.768 |
| XGBoost | 3 | 2430.593 | 3223.990 | 0.189 | 25.107 |
| XGBoost | 4 | 2865.251 | 3478.547 | 0.414 | 26.774 |
| XGBoost | 5 | 2127.954 | 2774.919 | 0.212 | 32.218 |

## 4. Ansatz 3 – SSL (Kap. 7.4)

**Ø-Metriken über 5 Folds**

| model | MAE_mean | RMSE_mean | R2_mean | MAPE_mean |
| --- | --- | --- | --- | --- |
| RF_SSL(labels+pseudo) | 2368.564 | 3104.188 | 0.438 | 27.042 |
| RF_supervised(only_labels) | 2331.575 | 3132.980 | 0.424 | 26.692 |

**Je Fold**

| model | fold | MAE | RMSE | R2 | MAPE | opt_q | n_pseudo |
| --- | --- | --- | --- | --- | --- | --- | --- |
| RF_SSL | 1 | 2569.668 | 3326.830 | 0.494 | 27.579 | 30.000 | 3327.000 |
| RF_SSL | 2 | 2160.649 | 3452.335 | 0.556 | 22.119 | 40.000 | 3920.000 |
| RF_SSL | 3 | 2278.233 | 2743.402 | 0.413 | 23.824 | 20.000 | 2591.000 |
| RF_SSL | 4 | 2771.659 | 3304.171 | 0.472 | 26.488 | 30.000 | 3285.000 |
| RF_SSL | 5 | 2062.613 | 2694.201 | 0.257 | 35.201 | 40.000 | 3955.000 |
| RF_supervised | 1 | 2447.138 | 3353.076 | 0.486 | 27.280 | nan | nan |
| RF_supervised | 2 | 2144.922 | 3526.457 | 0.537 | 21.304 | nan | nan |
| RF_supervised | 3 | 2311.308 | 2849.144 | 0.367 | 24.525 | nan | nan |
| RF_supervised | 4 | 2700.002 | 3169.737 | 0.514 | 26.678 | nan | nan |
| RF_supervised | 5 | 2054.503 | 2766.486 | 0.217 | 33.675 | nan | nan |

**SSL-Diagnose: Schwellwert-Sweep**

| q | MAE | RMSE | R2 | MAPE | n_pseudo_cum |
| --- | --- | --- | --- | --- | --- |
| 0.200 | 2348.700 | 3092.780 | 0.440 | 26.830 | 3486.200 |
| 0.300 | 2378.080 | 3119.900 | 0.430 | 27.040 | 4209.000 |
| 0.400 | 2494.870 | 3360.170 | 0.340 | 27.870 | 4628.800 |

## 5. Gesamtvergleich (Kap. 7.5)

| approach | model | MAE_mean | RMSE_mean | R2_mean | MAPE_mean |
| --- | --- | --- | --- | --- | --- |
| 1_baseline | Ridge | 2619.008 | 3772.815 | 0.160 | 30.495 |
| 1_baseline | RandomForest | 2693.224 | 3682.640 | 0.234 | 30.929 |
| 1_baseline | XGBoost | 2849.759 | 4032.107 | 0.065 | 31.538 |
| 2_delta | Ridge | 2115.568 | 2870.789 | 0.513 | 23.523 |
| 2_delta | RandomForest | 2152.438 | 2884.501 | 0.510 | 24.414 |
| 2_delta | XGBoost | 2449.034 | 3375.646 | 0.335 | 26.653 |
| 3_ssl | RF_SSL(labels+pseudo) | 2368.564 | 3104.188 | 0.438 | 27.042 |
| 3_ssl | RF_supervised(only_labels) | 2331.575 | 3132.980 | 0.424 | 26.692 |
