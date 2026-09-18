"""
predict_qsar_baseline.py
========================
Batch prediction of pIC50 using the pre-trained QSAR baseline model
(RandomForestRegressor on Morgan fingerprints, radius=2, 2048 bits).

Usage
-----
    python scripts/predict_qsar_baseline.py

Environment variables (all optional)
-------------------------------------
    MODEL_PATH   Path to the saved joblib model.
                 Default: outputs/qsar_baseline_model.joblib
    INPUT_CSV    Path to the input CSV with a SMILES column.
                 Default: Base de datos curada vf.csv
    OUTPUT_CSV   Path for the output CSV with the pIC50_pred column added.
                 Default: outputs/pred_base_curada_pic50.csv

SMILES column selection (in order of preference)
-------------------------------------------------
    1. "SMILE CANONICO"
    2. "Smiles"
    If neither is present the script exits with a clear error.

Invalid SMILES are written as NaN in the pIC50_pred column.
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd

# ── fingerprint parameters (must match training) ──────────────────────────────
FP_RADIUS = 2
FP_NBITS = 2048

# ── paths (overridable via environment) ───────────────────────────────────────
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join("outputs", "qsar_baseline_model.joblib"))
INPUT_CSV = os.environ.get("INPUT_CSV", "Base de datos curada vf.csv")
OUTPUT_CSV = os.environ.get("OUTPUT_CSV", os.path.join("outputs", "pred_base_curada_pic50.csv"))

# SMILES columns to try, in order of preference
_SMILES_CANDIDATES = ["SMILE CANONICO", "Smiles"]


def _load_csv(path: str) -> pd.DataFrame:
    """Load CSV with automatic delimiter detection (comma first, then semicolon)."""
    df = pd.read_csv(path, sep=",", low_memory=False)
    if df.shape[1] == 1:
        # Single column usually means wrong delimiter – retry with semicolon
        df = pd.read_csv(path, sep=";", low_memory=False)
    return df


def _detect_smiles_column(df: pd.DataFrame) -> str:
    """Return the first matching SMILES column name, or raise ValueError."""
    for col in _SMILES_CANDIDATES:
        if col in df.columns:
            return col
    raise ValueError(
        f"No SMILES column found in {INPUT_CSV}. "
        f"Expected one of {_SMILES_CANDIDATES}. "
        f"Available columns: {list(df.columns)}"
    )


def _smiles_to_fp(smiles_list: list) -> tuple:
    """Compute Morgan fingerprints; return (X, mask) matching training conventions."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    valid = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            valid.append(False)
            fps.append(None)
        else:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=FP_RADIUS, nBits=FP_NBITS)
            fps.append(list(fp))
            valid.append(True)

    mask = np.array(valid, dtype=bool)
    X = np.array([fp for fp, v in zip(fps, valid) if v], dtype=np.uint8)
    return X, mask


def main():
    # ── 1. validate inputs ────────────────────────────────────────────────────
    if not os.path.isfile(MODEL_PATH):
        print(
            f"ERROR: model not found at '{MODEL_PATH}'. "
            "Train the model first with scripts/train_qsar_baseline.py "
            "or set MODEL_PATH to a valid path.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.isfile(INPUT_CSV):
        print(f"ERROR: input CSV not found at '{INPUT_CSV}'.", file=sys.stderr)
        sys.exit(1)

    # ── 2. load data ──────────────────────────────────────────────────────────
    df = _load_csv(INPUT_CSV)
    smiles_col = _detect_smiles_column(df)

    # ── 3. load model ─────────────────────────────────────────────────────────
    model = joblib.load(MODEL_PATH)

    # ── 4. compute fingerprints ───────────────────────────────────────────────
    X, mask = _smiles_to_fp(df[smiles_col].astype(str).tolist())

    # ── 5. predict (NaN for invalid SMILES) ───────────────────────────────────
    preds = np.full(len(df), fill_value=np.nan, dtype=float)
    if mask.any():
        preds[mask] = model.predict(X)

    # ── 6. write output ───────────────────────────────────────────────────────
    out_dir = os.path.dirname(OUTPUT_CSV)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    df_out = df.copy()
    df_out["pIC50_pred"] = preds
    df_out.to_csv(OUTPUT_CSV, index=False)

    # ── 7. summary ────────────────────────────────────────────────────────────
    n_total = len(df)
    n_valid = int(mask.sum())
    n_invalid = n_total - n_valid
    print(f"Input  : {INPUT_CSV}  ({n_total} rows, SMILES column: '{smiles_col}')")
    print(f"Model  : {MODEL_PATH}")
    print(f"Valid SMILES  : {n_valid}")
    print(f"Invalid SMILES: {n_invalid}")
    print(f"Output : {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
