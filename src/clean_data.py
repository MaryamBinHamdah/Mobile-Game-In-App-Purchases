"""Stage 3: clean the expanded project dataset and save an audit summary.

This script demonstrates cleaning on generated data. It does not turn generated
rows into real observations. Run: python src/clean_data.py
"""
from __future__ import annotations

import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ZIP_FILE = ROOT / "data" / "02_expanded_dirty_data.zip"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_DIR = ROOT / "outputs" / "data_quality"
TARGET_LABELS = {"minnow": "Minnow", "dolphin": "Dolphin", "whale": "Whale"}


def read_project_data() -> pd.DataFrame:
    if not ZIP_FILE.exists():
        raise FileNotFoundError(f"Could not find {ZIP_FILE}")
    with zipfile.ZipFile(ZIP_FILE) as archive:
        csv_files = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if not csv_files:
            raise ValueError("The dataset archive contains no CSV file.")
        with archive.open(csv_files[0]) as file:
            return pd.read_csv(file)


def clean_duration(value):
    """Parse minutes and convert explicit seconds values into minutes."""
    if pd.isna(value):
        return np.nan
    text = str(value).strip().lower()
    match = re.fullmatch(r"([-+]?\d*\.?\d+)\s*(sec|seconds?)", text)
    if match:
        return float(match.group(1)) / 60
    text = text.replace("minutes", "").replace("minute", "").replace("min", "").strip()
    return pd.to_numeric(text, errors="coerce")


def clean_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    before = {"rows": int(len(raw)), "columns": int(raw.shape[1]),
              "missing_cells": int(raw.isna().sum().sum()),
              "exact_duplicates": int(raw.duplicated().sum())}
    data = raw.drop_duplicates().copy()
    rows_after_dedup = int(len(data))

    text_columns = data.select_dtypes(include=["object", "string"]).columns.tolist()
    for column in text_columns:
        data[column] = data[column].astype("string").str.strip()
        data[column] = data[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

    # Canonical labels for common variants observed in this project file.
    maps = {
        "Device": {"ios": "iOS", "IOS": "iOS", "android": "Android"},
        "GameGenre": {"battleroyale": "Battle Royale", "battle royale": "Battle Royale"},
        "SpendingSegment": TARGET_LABELS,
    }
    for column, mapping in maps.items():
        if column in data:
            data[column] = data[column].astype("string").str.strip()
            lower_map = {str(k).lower(): v for k, v in mapping.items()}
            data[column] = data[column].map(
                lambda value: lower_map.get(str(value).strip().lower(), value) if pd.notna(value) else value
            )
    for column in ["Gender", "PaymentMethod", "AcquisitionChannel", "Country"]:
        if column in data:
            data[column] = data[column].replace({"m": "Male", "f": "Female"})

    # Normalize mixed text/numeric values before range checks.
    if "AverageSessionLength" in data:
        data["AverageSessionLength"] = data["AverageSessionLength"].map(clean_duration)
    if "InAppPurchaseAmount" in data:
        amount = data["InAppPurchaseAmount"].astype("string").str.replace(r"[$,]", "", regex=True)
        data["InAppPurchaseAmount"] = pd.to_numeric(amount, errors="coerce")

    numeric_columns = [
        "Age", "SessionCount", "AverageSessionLength", "FirstPurchaseDaysAfterInstall",
        "ActiveDays30d", "PlayerLevel30d", "AdsWatched30d", "SocialActions30d",
        "PurchaseCount30d", "WinRate",
    ]
    for column in numeric_columns:
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")

    # Invalid measurements become missing; do not silently take absolute values.
    invalid_counts = {}
    rules = {
        "Age": lambda s: ~s.between(13, 100),
        "SessionCount": lambda s: s < 0,
        "AverageSessionLength": lambda s: s < 0,
        "FirstPurchaseDaysAfterInstall": lambda s: s < 0,
        "ActiveDays30d": lambda s: ~s.between(0, 30),
        "PlayerLevel30d": lambda s: s < 0,
        "AdsWatched30d": lambda s: s < 0,
        "SocialActions30d": lambda s: s < 0,
        "PurchaseCount30d": lambda s: s < 0,
        "WinRate": lambda s: ~s.between(0, 1),
        "InAppPurchaseAmount": lambda s: s < 0,
    }
    for column, invalid_rule in rules.items():
        if column in data:
            invalid = data[column].notna() & invalid_rule(data[column])
            invalid_counts[column] = int(invalid.sum())
            data.loc[invalid, column] = np.nan

    # Parse dates. Unparseable and impossible future values are treated as missing.
    date_columns = [c for c in ["InstallDate", "LastPurchaseDate"] if c in data]
    date_invalid_counts = {}
    for column in date_columns:
        parsed = pd.to_datetime(data[column], errors="coerce", format="mixed", dayfirst=False, utc=True)
        date_invalid_counts[column] = int((data[column].notna() & parsed.isna()).sum())
        data[column] = parsed
    if "InstallDate" in data:
        future = data["InstallDate"] > pd.Timestamp.now(tz="UTC")
        date_invalid_counts["future_install_dates"] = int(future.sum())
        data.loc[future, "InstallDate"] = pd.NaT
    if {"InstallDate", "LastPurchaseDate"}.issubset(data.columns):
        reversed_dates = data["InstallDate"].notna() & data["LastPurchaseDate"].notna() & (
            data["InstallDate"] > data["LastPurchaseDate"]
        )
        date_invalid_counts["install_after_purchase"] = int(reversed_dates.sum())
        data.loc[reversed_dates, "InstallDate"] = pd.NaT

    missing_before_imputation = int(data.isna().sum().sum())
    # Simple descriptive-data imputation. In modeling, imputation is fitted only
    # within a training-fold pipeline; never learn these medians from the test set.
    for column in numeric_columns + (["InAppPurchaseAmount"] if "InAppPurchaseAmount" in data else []):
        if column in data and data[column].isna().any():
            data[column] = data[column].fillna(data[column].median())
    for column in data.select_dtypes(include=["object", "string"]).columns:
        if column == "UserID":
            data[column] = data[column].fillna("Unknown")
        elif data[column].isna().any():
            mode = data[column].mode(dropna=True)
            data[column] = data[column].fillna(mode.iloc[0] if len(mode) else "Unknown")
    for column in date_columns:
        if data[column].isna().any():
            # Median timestamps are only used to make a complete instructional
            # cleaning export; the date fields are excluded from model inputs.
            data[column] = data[column].fillna(data[column].dropna().median())

    after = {"rows": int(len(data)), "columns": int(data.shape[1]),
             "missing_cells": int(data.isna().sum().sum()),
             "exact_duplicates": int(data.duplicated().sum())}
    audit = {
        "before": before,
        "after": after,
        "rows_after_duplicate_removal": rows_after_dedup,
        "missing_cells_before_imputation": missing_before_imputation,
        "invalid_numeric_values_by_column": invalid_counts,
        "date_issues": date_invalid_counts,
        "target_counts_after_cleaning": data["SpendingSegment"].value_counts().to_dict()
        if "SpendingSegment" in data else {},
        "important_note": "The expanded dataset contains generated records. The cleaning exercise does not make them real-world observations.",
    }
    return data, audit


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    raw = read_project_data()
    clean, audit = clean_data(raw)
    if clean.isna().sum().sum() != 0:
        raise AssertionError("Missing values remain after the documented imputation step.")
    if clean.duplicated().any():
        raise AssertionError("Exact duplicate rows remain after cleaning.")
    clean.to_csv(OUTPUT_DIR / "cleaned_players.csv", index=False)
    (REPORT_DIR / "cleaning_audit.json").write_text(json.dumps(audit, indent=2, default=str), encoding="utf-8")
    print(f"Rows: {audit['before']['rows']:,} -> {audit['after']['rows']:,}")
    print(f"Columns: {audit['before']['columns']} -> {audit['after']['columns']}")
    print(f"Exact duplicates removed: {audit['before']['exact_duplicates']:,}")
    print(f"Missing cells before imputation: {audit['missing_cells_before_imputation']:,}")
    print(f"Missing cells after cleaning: {audit['after']['missing_cells']:,}")
    print(f"Saved: {OUTPUT_DIR / 'cleaned_players.csv'}")
    print("Reminder: these expanded records are synthetic and are not additional measured players.")


if __name__ == "__main__":
    main()
