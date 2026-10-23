"""Shared data definitions so every notebook builds labels and splits the same way."""
import os, ast, time
import numpy as np
import pandas as pd
import requests

DATA_DIR = os.path.join("data", "ptb-xl")
BASE_URL = "https://physionet-open.s3.amazonaws.com/ptb-xl/1.0.3/"
SPLIT_FILE = os.path.join("data", "splits.csv")
RESULTS_DIR = "results"

SUPER, ABN = ["NORM", "MI", "STTC", "CD", "HYP"], ["MI", "STTC", "CD", "HYP"]
SPLITS = ["train", "eyeball", "blackbox", "test"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_SETS = {"A_12lead": LEADS, "B_limb": LEADS[:6], "C_leadI": ["I"]}
CONFIGS = list(LEAD_SETS)


def fetch_ptbxl(rel_paths, data_dir=DATA_DIR, retries=5):
    """Download individual PTB-XL files (paths relative to the dataset root) that aren't on disk yet.
    Lets a fresh Colab runtime get just the label CSVs or a single record without the full 1.6 GB."""
    for rel in rel_paths:
        dest = os.path.join(data_dir, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            continue
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        for attempt in range(retries):
            try:
                r = requests.get(BASE_URL + rel, timeout=(10, 30))
                r.raise_for_status()
                with open(dest + ".part", "wb") as fh:
                    fh.write(r.content)
                os.replace(dest + ".part", dest)
                break
            except Exception:
                if attempt == retries - 1:
                    raise
                time.sleep(2 ** attempt)


def load_labels(data_dir=DATA_DIR):
    """One row per usable recording (indexed by ecg_id): patient_id, filename_lr, scp_codes,
    the 5 superclass columns, `abnormal`, and `split` (from data/splits.csv)."""
    fetch_ptbxl(["ptbxl_database.csv", "scp_statements.csv"], data_dir)
    db = pd.read_csv(os.path.join(data_dir, "ptbxl_database.csv"), index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(os.path.join(data_dir, "scp_statements.csv"), index_col=0)
    diag_map = scp.loc[scp.diagnostic == 1, "diagnostic_class"].to_dict()
    supers = db.scp_codes.apply(lambda codes: {diag_map[c] for c in codes if c in diag_map})
    for c in SUPER:
        db[c] = supers.apply(lambda s: int(c in s))
    db["abnormal"] = (db[ABN].sum(axis=1) > 0).astype(int)
    df = db.loc[db[SUPER].sum(axis=1) > 0].copy()

    assert os.path.exists(SPLIT_FILE), "data/splits.csv missing: run 02_baselines.ipynb first"
    df["split"] = pd.read_csv(SPLIT_FILE, index_col="ecg_id").loc[df.index, "split"]
    return df


def load_predictions(name, df):
    """Load results/<name>.npz and align it to df's row order. Returns probabilities as a Series."""
    z = np.load(os.path.join(RESULTS_DIR, f"{name}.npz"), allow_pickle=False)
    p = pd.Series(z["p"], index=z["ecg_id"]).loc[df.index]
    assert (pd.Series(z["split"], index=z["ecg_id"]).loc[df.index] == df.split).all(), \
        f"{name}: split in predictions file does not match data/splits.csv"
    return p
