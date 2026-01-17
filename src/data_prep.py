from __future__ import annotations
from pathlib import Path
import pandas as pd

REQUIRED_COLS = [
    "patient_id",
    "age",
    "gender",
    "egfr",
    "insulin_user",
    "current_medications",
    "comorbidities",
]

def insulin_flag(val) -> bool:
    s = str(val).strip().lower()
    return s in ["true", "1", "1.0", "yes", "y"]

def clean(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()

def make_patient_card(row: pd.Series) -> str:
    return (
        "PATIENT:\n"
        f"patient_id: {clean(row['patient_id'])}\n"
        f"age: {clean(row['age'])}\n"
        f"gender: {clean(row['gender'])}\n"
        f"egfr: {clean(row['egfr'])}\n"
        f"insulin_user: {clean(row['insulin_user'])}\n"
        f"current_medications: {clean(row['current_medications'])}\n"
        f"comorbidities: {clean(row['comorbidities'])}\n"
    )

def build_patient_cards(in_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(in_csv)

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()
    df["insulin_user_bool"] = df["insulin_user"].apply(insulin_flag)
    df["egfr_num"] = pd.to_numeric(df["egfr"], errors="coerce")

    df["patient_card"] = df.apply(make_patient_card, axis=1)

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    return df
