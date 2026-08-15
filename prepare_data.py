"""Prepare the UCI Steel Plates Faults data for this assignment."""

from pathlib import Path

import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_DIR / "Data" / "Faults.NNA"
VARIABLES_PATH = PROJECT_DIR / "Data" / "Faults27x7_var"
OUTPUT_PATH = PROJECT_DIR / "Data" / "steel_plates_faults_clean.csv"


def main() -> None:
    """Create a clean CSV with one human-readable target column."""
    column_names = VARIABLES_PATH.read_text(encoding="utf-8").splitlines()
    data = pd.read_csv(RAW_DATA_PATH, sep=r"\s+", header=None, names=column_names)

    feature_columns = column_names[:27]
    target_columns = column_names[27:]
    if len(target_columns) != 7:
        raise ValueError("Expected seven one-hot fault-label columns.")
    if not data[target_columns].sum(axis=1).eq(1).all():
        raise ValueError("Every row must have exactly one fault label.")

    cleaned_data = data[feature_columns].copy()
    cleaned_data["fault_type"] = data[target_columns].idxmax(axis=1)
    cleaned_data.to_csv(OUTPUT_PATH, index=False)
    print(f"Created {OUTPUT_PATH}")
    print(f"Rows: {len(cleaned_data)} | Features: {len(feature_columns)}")
    print(cleaned_data["fault_type"].value_counts().sort_index().to_string())


if __name__ == "__main__":
    main()
