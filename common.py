"""Shared data definitions so every notebook builds labels and splits the same way."""
import os, ast
import numpy as np
import pandas as pd

DATA_DIR = os.path.join("data", "ptb-xl")
SPLIT_FILE = os.path.join("data", "splits.csv")
RESULTS_DIR = "results"

SUPER, ABN = ["NORM", "MI", "STTC", "CD", "HYP"], ["MI", "STTC", "CD", "HYP"]
SPLITS = ["train", "eyeball", "blackbox", "test"]
LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
LEAD_SETS = {"A_12lead": LEADS, "B_limb": LEADS[:6], "C_leadI": ["I"]}
CONFIGS = list(LEAD_SETS)


def load_labels(data_dir=DATA_DIR):
    """One row per usable recording (indexed by ecg_id): patient_id, filename_lr, scp_codes,
    the 5 superclass columns, `abnormal`, and `split` (from data/splits.csv)."""
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
