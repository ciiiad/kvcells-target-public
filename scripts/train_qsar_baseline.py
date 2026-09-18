"""
train_qsar_baseline.py
======================
Minimal, robust QSAR baseline for KB-cells cytotoxicity (pIC50).

Pipeline:
  1. Read kb_dataset_IC50_routeA_clean.csv (runs cleaning script if missing).
  2. Compute Morgan fingerprints (radius=2, nBits=2048) with RDKit.
  3. Drop rows where RDKit cannot parse SMILES.
  4. Train/test split (80/20, random_state=42).
  5. Train RandomForestRegressor (n_estimators=300, random_state=42, n_jobs=-1).
  6. Evaluate: R², RMSE, MAE on test set.
  7. Save metrics → outputs/qsar_baseline_metrics.json
  8. Save scatter plot → outputs/qsar_pred_vs_true.png
  9. Save model → outputs/qsar_baseline_model.joblib
"""

import json
import os
import subprocess
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ── paths ──────────────────────────────────────────────────────────────────────
CLEAN_CSV = os.environ.get("CLEAN_CSV", "kb_dataset_IC50_routeA_clean.csv")
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")
METRICS_JSON = os.path.join(OUTPUT_DIR, "qsar_baseline_metrics.json")
PLOT_PNG = os.path.join(OUTPUT_DIR, "qsar_pred_vs_true.png")
MODEL_JOBLIB = os.path.join(OUTPUT_DIR, "qsar_baseline_model.joblib")

# ── hyperparameters ────────────────────────────────────────────────────────────
FP_RADIUS = 2
FP_NBITS = 2048
N_ESTIMATORS = 300
RANDOM_STATE = 42
TEST_SIZE = 0.2


def ensure_clean_csv():
    if not os.path.isfile(CLEAN_CSV):
        print(f"{CLEAN_CSV} not found – running cleaning script …")
        script = os.path.join(os.path.dirname(__file__), "clean_kb_ic50_pic50.py")
        result = subprocess.run(
            [sys.executable, script],
            env={**os.environ, "CLEAN_OUTPUT": CLEAN_CSV},
            capture_output=True,
            text=True,
        )
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr, file=sys.stderr)
        if result.returncode != 0 or not os.path.isfile(CLEAN_CSV):
            print("ERROR: cleaning script failed.", file=sys.stderr)
            sys.exit(1)


def smiles_to_fp(smiles_series):
    """Return (X, mask) where X is the fingerprint matrix and mask marks valid rows."""
    from rdkit import Chem
    from rdkit.Chem import AllChem

    fps = []
    valid = []
    for smi in smiles_series:
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
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── 1. load data ───────────────────────────────────────────────────────────
    ensure_clean_csv()
    df = pd.read_csv(CLEAN_CSV)
    print(f"Loaded {len(df)} rows from {CLEAN_CSV}")

    # ── 2–3. fingerprints + drop invalid ──────────────────────────────────────
    print("Computing Morgan fingerprints …")
    X, mask = smiles_to_fp(df["Smiles"])
    n_dropped = (~mask).sum()
    if n_dropped:
        print(f"  Dropped {n_dropped} rows (RDKit could not parse SMILES)")
    df_valid = df[mask].reset_index(drop=True)
    y = df_valid["pIC50"].values
    print(f"  {len(df_valid)} molecules with valid fingerprints")

    if len(df_valid) < 10:
        print("ERROR: too few valid molecules to train/test split.", file=sys.stderr)
        sys.exit(1)

    # ── 4. train/test split ───────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}  Test: {len(X_test)}")

    # ── 5. train model ────────────────────────────────────────────────────────
    print(f"Training RandomForestRegressor (n_estimators={N_ESTIMATORS}) …")
    rf = RandomForestRegressor(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    # ── 6. evaluate ───────────────────────────────────────────────────────────
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    mae = float(mean_absolute_error(y_test, y_pred))

    print("\n── Test-set metrics ──────────────────────────────────")
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    print(f"  MAE  : {mae:.4f}")
    print("─────────────────────────────────────────────────────")

    # ── 7. save metrics ───────────────────────────────────────────────────────
    metrics = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_dropped_invalid_smiles": int(n_dropped),
        "fp_radius": FP_RADIUS,
        "fp_nbits": FP_NBITS,
        "n_estimators": N_ESTIMATORS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "R2": round(r2, 6),
        "RMSE": round(rmse, 6),
        "MAE": round(mae, 6),
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {METRICS_JSON}")

    # ── 8. scatter plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(y_test, y_pred, alpha=0.4, edgecolors="none", s=20, color="steelblue")
    lims = [min(y_test.min(), y_pred.min()) - 0.5, max(y_test.max(), y_pred.max()) + 0.5]
    ax.plot(lims, lims, "r--", linewidth=1, label="ideal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual pIC50")
    ax.set_ylabel("Predicted pIC50")
    ax.set_title(
        f"QSAR baseline – KB cells (n_test={len(X_test)})\n"
        f"R²={r2:.3f}  RMSE={rmse:.3f}  MAE={mae:.3f}"
    )
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOT_PNG, dpi=150)
    plt.close(fig)
    print(f"Plot saved → {PLOT_PNG}")

    # ── 9. save model ─────────────────────────────────────────────────────────
    joblib.dump(rf, MODEL_JOBLIB)
    print(f"Model saved → {MODEL_JOBLIB}")


if __name__ == "__main__":
    main()
