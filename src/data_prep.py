"""
Datenaufbereitung (Kap. 4.2-4.4 / 5.2).

Erzeugt zwei Datensätze mit IDENTISCHEM Feature-Schema:
  1. labeled.csv            -> ~112 Fälle mit bestätigtem finalem Reparaturpreis
                               (Grundlage für Ansatz 1 & 2)
  2. unlabeled_pool_full.csv -> alle Kostenvoranschläge (KVs) mit Fahrzeug-Features,
                               OHNE finalen Preis (Grundlage für Ansatz 3)

Quellen:
  - workshops_final_prices.xlsx : finale Preise + Features (gelabelt)
  - offers_with_estimates.xlsx  : alle KVs (Engine Repair Price) + Lead ID
  - Customer-*.xlsx             : vollständiger Kundenexport mit Fahrzeug-Features

Hinweis Salesforce-IDs: Der Offers-Export nutzt 18-stellige, der Customer-Export
15-stellige Lead-IDs. Gejoint wird auf den ersten 15 Zeichen (lead15).
"""
from __future__ import annotations
import re
import sys
import warnings
import pandas as pd

sys.path.append(str(__import__("pathlib").Path(__file__).resolve().parents[1]))
import config as C  # noqa: E402

warnings.filterwarnings("ignore")

# Salesforce-Spalten -> internes Schema
COLUMN_MAP = {
    "Lead ID": "lead_id",
    "Brand": "brand",
    "Model": "model",
    "Year Of Construction": "year_of_construction",
    "Mileage": "mileage",
    "Car from country?": "car_from_country",
    "HSN": "hsn",
    "KW": "kw",
    "Vehicle ready to drive": "vehicle_ready_to_drive",
    "Fuel Art": "fuel_type",
    "Motor Code": "motor_code",
    "Condition": "damage_type",
    "Engine Repair Price": "price_estimation",
    "final_price": "final_price",
}

# Markennamen, die in den Rohdaten uneinheitlich sind, werden zusammengeführt.
BRAND_NORMALISE = {
    "VW": "Volkswagen",
    "Vw": "Volkswagen",
    "Mercedes Benz": "Mercedes-Benz",
    "Mercedes": "Mercedes-Benz",
}


def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df[[c for c in df.columns if "Unnamed" not in str(c)]].copy()
    df.columns = (
        pd.Index(df.columns).astype(str)
        .str.replace("\n", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    return df


def parse_mileage(val):
    if pd.isna(val):
        return None
    s = re.sub(r"[^\d,.]", "", str(val).strip()).replace(".", "").replace(",", "")
    try:
        return float(s) if s else None
    except ValueError:
        return None


def parse_kw(val):
    if pd.isna(val):
        return None
    m = re.match(r"^(\d+(?:[.,]\d+)?)", str(val).strip())
    return float(m.group(1).replace(",", ".")) if m else None


def _normalise(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={k: v for k, v in COLUMN_MAP.items() if k in df.columns})
    if "brand" in df.columns:
        df["brand"] = (
            df["brand"].astype(str).str.strip().replace(BRAND_NORMALISE)
        )
    for col in ("mileage", "kw"):
        if col in df.columns:
            df[col] = df[col].apply(parse_mileage if col == "mileage" else parse_kw)
    for col in ("year_of_construction", "hsn", "price_estimation", "final_price"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if "year_of_construction" in df.columns:
        df["vehicle_age"] = C.REFERENCE_YEAR - df["year_of_construction"]
    return df


def build_labeled() -> pd.DataFrame:
    """Gelabelter Datensatz aus den finalen Werkstattpreisen."""
    w = _clean_cols(pd.read_excel(C.WORKSHOPS_FINAL_XLSX, header=2))
    w = w.rename(columns={"Finaler Preis": "final_price"})
    w = _normalise(w)
    w["lead15"] = w["lead_id"].astype(str).str.strip().str[:15]
    df = w.dropna(subset=["final_price"]).copy()
    keep = C.FEATURES + ["final_price", "lead15"]
    df = df[[c for c in keep if c in df.columns]]
    # nur sinnvolle Preise
    df = df[df["final_price"] > 0]
    return df.reset_index(drop=True)


def build_unlabeled_pool(labeled_lead15: set[str]) -> pd.DataFrame:
    """Alle KVs mit Fahrzeug-Features; gelabelte Leads werden ausgeschlossen."""
    cust = _clean_cols(pd.read_excel(C.CUSTOMER_FULL_XLSX, header=8))
    off = _clean_cols(pd.read_excel(C.OFFERS_XLSX, header=9))
    off["lead15"] = off["Lead ID"].astype(str).str.strip().str[:15]
    cust["lead15"] = cust["Lead ID"].astype(str).str.strip().str[:15]
    # KV-Preis aus Offers + Fahrzeug-Features aus Customer
    off = off.rename(columns={"Engine Repair Price": "price_estimation"})
    off["price_estimation"] = pd.to_numeric(off["price_estimation"], errors="coerce")
    merged = off[["lead15", "price_estimation"]].merge(
        cust, on="lead15", how="inner", suffixes=("", "_cust")
    )
    merged = _normalise(merged)
    # gelabelte Fälle entfernen -> echter Unlabeled-Pool
    merged = merged[~merged["lead15"].isin(labeled_lead15)].copy()
    keep = C.FEATURES + ["lead15"]
    merged = merged[[c for c in keep if c in merged.columns]]
    merged = merged[merged["price_estimation"] > 0]
    merged = merged.dropna(subset=["brand"])
    return merged.reset_index(drop=True)


def main():
    labeled = build_labeled()
    labeled.to_csv(C.LABELED_CSV, index=False)
    print(f"[data_prep] labeled.csv: {labeled.shape}  "
          f"({labeled['final_price'].notna().sum()} finale Preise)")

    pool = build_unlabeled_pool(set(labeled["lead15"]))
    pool.to_csv(C.UNLABELED_POOL_CSV, index=False)
    print(f"[data_prep] unlabeled_pool_full.csv: {pool.shape}  "
          f"({pool['lead15'].nunique()} eindeutige Leads)")


if __name__ == "__main__":
    main()
