"""
01_data_cleaning.py
--------------------
Business Bank Customer Churn Project — Step 1: Data Cleaning

Goal:
    The bank actually receives customer data as TWO separate messy Excel
    sheets (Customer_Info and Account_Info) that need to be merged and
    cleaned before any analysis can happen. This script simulates that
    real-world data engineering step.

What this script does:
    1. Loads the two raw sheets from Bank_Churn_Messy.xlsx
    2. Fixes data type issues (currency strings -> float, Yes/No -> 1/0)
    3. Removes duplicate rows
    4. Merges the two sheets into a single customer-level table on CustomerId
    5. Validates the cleaned output against the "ground truth" clean file
       (Bank_Churn.csv) to prove the cleaning logic is correct
    6. Saves the final clean dataset to data/processed/bank_churn_clean.csv

Run:
    python src/01_data_cleaning.py
"""

import pandas as pd
import numpy as np
import re

RAW_XLSX = "data/raw/Bank_Churn_Messy.xlsx"
REFERENCE_CSV = "data/raw/Bank_Churn.csv"   # used only to validate our cleaning
OUTPUT_CSV = "data/processed/bank_churn_clean.csv"


def clean_currency(series: pd.Series) -> pd.Series:
    """Strip currency symbols/commas and convert to float."""
    return (
        series.astype(str)
        .str.replace(r"[€$,]", "", regex=True)
        .str.strip()
        .astype(float)
    )


def clean_sentinel_missing(series: pd.Series, sentinel_value: float = -999999) -> pd.Series:
    """Some records use -999999 as a 'missing value' sentinel instead of a
    true NaN (a common real-world data-entry quirk). Convert those to NaN."""
    numeric = clean_currency(series)
    numeric = numeric.where(numeric != sentinel_value, np.nan)
    return numeric


def clean_yes_no(series: pd.Series) -> pd.Series:
    """Convert Yes/No (and 1/0 already-clean values) to integer 0/1."""
    mapping = {"yes": 1, "no": 0, "y": 1, "n": 0}
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
        .fillna(series)  # in case some rows are already 0/1
        .astype(int)
    )


def main():
    print("Loading raw messy workbook...")
    customer_info = pd.read_excel(RAW_XLSX, sheet_name="Customer_Info")
    account_info = pd.read_excel(RAW_XLSX, sheet_name="Account_Info")

    print(f"  Customer_Info raw shape: {customer_info.shape}")
    print(f"  Account_Info raw shape:  {account_info.shape}")

    # --- Clean Customer_Info ---
    customer_info = customer_info.drop_duplicates()

    # Geography has inconsistent labels for the same country (e.g. "France",
    # "French", "FRA") that need to be standardised before analysis
    geography_map = {
        "france": "France", "french": "France", "fra": "France",
        "germany": "Germany", "german": "Germany", "ger": "Germany",
        "spain": "Spain", "spanish": "Spain", "esp": "Spain",
    }
    raw_geo_values = customer_info["Geography"].value_counts().to_dict()
    customer_info["Geography"] = (
        customer_info["Geography"].astype(str).str.strip().str.lower().map(geography_map)
    )
    print(f"  Standardised Geography labels. Raw variants found: {raw_geo_values}")

    # EstimatedSalary uses a -€999999 sentinel for missing values in a few
    # rows; convert those to real NaN, then impute with the column median
    customer_info["EstimatedSalary"] = clean_sentinel_missing(customer_info["EstimatedSalary"])
    salary_missing = customer_info["EstimatedSalary"].isna().sum()
    customer_info["EstimatedSalary"] = customer_info["EstimatedSalary"].fillna(
        customer_info["EstimatedSalary"].median()
    )

    # A few Age and Surname values are genuinely blank -> impute/flag
    age_missing = customer_info["Age"].isna().sum()
    customer_info["Age"] = customer_info["Age"].fillna(customer_info["Age"].median())
    customer_info["Age"] = customer_info["Age"].round().astype(int)
    customer_info["Surname"] = customer_info["Surname"].fillna("Unknown")

    print(f"  Imputed {salary_missing} sentinel-missing EstimatedSalary values with median")
    print(f"  Imputed {age_missing} missing Age values with median")

    # --- Clean Account_Info ---
    account_info = account_info.drop_duplicates()
    account_info["Balance"] = clean_currency(account_info["Balance"])
    account_info["HasCrCard"] = clean_yes_no(account_info["HasCrCard"])
    account_info["IsActiveMember"] = clean_yes_no(account_info["IsActiveMember"])

    print(f"  Customer_Info after de-dup: {customer_info.shape}")
    print(f"  Account_Info after de-dup:  {account_info.shape}")

    # Account_Info carries its own Tenure column too (duplicated field
    # across both sheets) - drop it from one side before merging so we
    # don't get Tenure_x / Tenure_y
    if "Tenure" in account_info.columns and "Tenure" in customer_info.columns:
        account_info = account_info.drop(columns=["Tenure"])

    # --- Merge on CustomerId ---
    merged = customer_info.merge(account_info, on="CustomerId", how="inner")
    merged = merged.drop_duplicates(subset="CustomerId")

    print(f"  Merged shape (before column reorder): {merged.shape}")

    # Reorder to match the reference schema
    col_order = [
        "CustomerId", "Surname", "CreditScore", "Geography", "Gender", "Age",
        "Tenure", "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary", "Exited",
    ]
    merged = merged[col_order]

    # --- Basic sanity checks ---
    assert merged["CustomerId"].is_unique, "CustomerId is not unique after cleaning!"
    assert merged.isna().sum().sum() == 0, "Nulls remain after cleaning!"
    assert merged["Exited"].isin([0, 1]).all()
    assert merged["Geography"].isin(["France", "Germany", "Spain"]).all(), "Unmapped geography label found!"
    assert merged["HasCrCard"].isin([0, 1]).all()
    assert merged["IsActiveMember"].isin([0, 1]).all()

    # --- Validate against reference clean CSV, if available ---
    try:
        reference = pd.read_csv(REFERENCE_CSV)
        reference = reference.sort_values("CustomerId").reset_index(drop=True)
        check = merged.sort_values("CustomerId").reset_index(drop=True)
        common_ids = set(reference.CustomerId) & set(check.CustomerId)
        print(f"  Rows matching reference on CustomerId: {len(common_ids)} / {len(reference)}")
        # spot check numeric columns are close
        ref_sub = reference[reference.CustomerId.isin(common_ids)].sort_values("CustomerId")
        chk_sub = check[check.CustomerId.isin(common_ids)].sort_values("CustomerId")
        for col in ["CreditScore", "Balance", "EstimatedSalary", "Exited"]:
            match_rate = np.isclose(
                ref_sub[col].values.astype(float), chk_sub[col].values.astype(float)
            ).mean()
            print(f"    {col}: {match_rate:.1%} match rate vs reference")
    except FileNotFoundError:
        print("  (Reference CSV not found — skipping validation step)")

    merged.to_csv(OUTPUT_CSV, index=False)
    print(f"\nClean dataset saved to {OUTPUT_CSV}  — final shape: {merged.shape}")


if __name__ == "__main__":
    main()
