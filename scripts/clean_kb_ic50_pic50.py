"""
clean_kb_ic50_pic50.py
======================
Conservative 8-step cleaning pipeline: KB-cells IC50 data → pIC50

Route A (quantitative QSAR baseline):
  1. Keep only Standard Type == IC50
  2. Keep only Standard Relation empty or = (strips surrounding single-quotes)
  3. Keep only Standard Units in {nM, uM}
  4. Require Standard Value is numeric and > 0
  5. Require Smiles is non-empty
  6. Convert to molar and compute pIC50 = -log10(M) using NumPy
  7. Sanity filter: keep pIC50 ∈ [3.0, 12.0]
  8. Deduplicate by Smiles using median pIC50

Output: kb_dataset_IC50_routeA_clean.csv
"""

import os
import sys

import numpy as np
import pandas as pd

INPUT = os.environ.get("CLEAN_INPUT", "kb_dataset_FIXED.csv")
OUTPUT = os.environ.get("CLEAN_OUTPUT", "kb_dataset_IC50_routeA_clean.csv")


def to_float_safe(x):
    try:
        if pd.isna(x):
            return None
        s = str(x).strip().strip('"').strip("'")
        if s == "" or s.lower() in ("none", "nan"):
            return None
        return float(s)
    except (ValueError, TypeError):
        return None


def unit_to_molar(value_series, unit_series):
    result = np.where(
        unit_series == "nM",
        value_series * 1e-9,
        np.where(unit_series == "uM", value_series * 1e-6, np.nan),
    )
    return result


def main():
    if not os.path.isfile(INPUT):
        print(f"ERROR: input file not found: {INPUT}", file=sys.stderr)
        sys.exit(1)

    print(f"Reading {INPUT} …")
    df = pd.read_csv(INPUT, sep=";", dtype=str, engine="python")
    total = len(df)
    print(f"  Total rows: {total}")

    # Normalise key text fields
    for col in ["Standard Type", "Standard Relation", "Standard Value", "Standard Units", "Smiles"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.strip('"')

    # Step 1 – IC50 only
    df = df[df["Standard Type"] == "IC50"].copy()
    print(f"  After IC50 filter: {len(df)}")

    # Step 2 – relation = '' or '='
    df["Standard Relation"] = df["Standard Relation"].str.replace("'", "", regex=False).str.strip()
    df = df[df["Standard Relation"].isin(["", "="])].copy()
    print(f"  After relation filter: {len(df)}")

    # Step 3 – units nM or uM
    df["Standard Units"] = df["Standard Units"].str.strip()
    df = df[df["Standard Units"].isin(["nM", "uM"])].copy()
    print(f"  After units filter: {len(df)}")

    # Step 4 – numeric value > 0
    df["_value"] = df["Standard Value"].apply(to_float_safe)
    df = df[df["_value"].notna() & (df["_value"] > 0)].copy()
    print(f"  After value filter: {len(df)}")

    # Step 5 – non-empty SMILES
    df["Smiles"] = df["Smiles"].astype(str).str.strip()
    df = df[df["Smiles"].notna() & (df["Smiles"] != "") & (df["Smiles"].str.lower() != "nan")].copy()
    print(f"  After SMILES filter: {len(df)}")

    # Step 6 – convert to molar, compute pIC50
    molar = unit_to_molar(df["_value"].values, df["Standard Units"].values)
    df["_molar"] = molar
    df = df[~np.isnan(molar) & (molar > 0)].copy()
    df["pIC50"] = -np.log10(df["_molar"])
    print(f"  After molar conversion: {len(df)}")

    # Step 7 – sanity range
    df = df[(df["pIC50"] >= 3.0) & (df["pIC50"] <= 12.0)].copy()
    print(f"  After pIC50 range filter: {len(df)}")

    # Step 8 – deduplicate by Smiles (median pIC50)
    agg_dict = {"pIC50": "median"}
    for col in ["Molecule ChEMBL ID", "Molecule Name"]:
        if col in df.columns:
            agg_dict[col] = "first"

    df_out = df.groupby("Smiles", as_index=False).agg(agg_dict)
    print(f"  After deduplication: {len(df_out)} unique SMILES")

    # Final column order
    cols_keep = [c for c in ["Molecule ChEMBL ID", "Molecule Name", "Smiles", "pIC50"] if c in df_out.columns]
    df_out = df_out[cols_keep].sort_values("pIC50", ascending=False).reset_index(drop=True)

    df_out.to_csv(OUTPUT, index=False)
    print(f"\nWrote {OUTPUT}  ({len(df_out)} rows)")


if __name__ == "__main__":
    main()
