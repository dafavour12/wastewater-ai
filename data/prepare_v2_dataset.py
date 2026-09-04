from pathlib import Path

import pandas as pd


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_FILE = BASE_DIR / "data" / "raw" / "water-treatment.data"
OUTPUT_DIR = BASE_DIR / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "wastewater_v2.csv"


# ---------------------------------------------------------------------
# Official UCI Water Treatment Plant column names
# ---------------------------------------------------------------------

UCI_COLUMNS = [
    "Q-E",
    "ZN-E",
    "PH-E",
    "DBO-E",
    "DQO-E",
    "SS-E",
    "SSV-E",
    "SED-E",
    "COND-E",
    "PH-P",
    "DBO-P",
    "SS-P",
    "SSV-P",
    "SED-P",
    "COND-P",
    "PH-D",
    "DBO-D",
    "DQO-D",
    "SS-D",
    "SSV-D",
    "SED-D",
    "COND-D",
    "PH-S",
    "DBO-S",
    "DQO-S",
    "SS-S",
    "SSV-S",
    "SED-S",
    "COND-S",
    "RD-DBO-P",
    "RD-SS-P",
    "RD-SED-P",
    "RD-DBO-S",
    "RD-DQO-S",
    "RD-DBO-G",
    "RD-DQO-G",
    "RD-SS-G",
    "RD-SED-G",
]


# ---------------------------------------------------------------------
# Selected variables for Wastewater AI V2
# ---------------------------------------------------------------------
#
# UCI variable       Wastewater AI V2 variable
#
# Q-E                flow_m3_day
# PH-E               influent_ph
# DBO-E              influent_bod5
# DQO-E              influent_cod
# SS-E               influent_tss
# PH-S               effluent_ph
# DBO-S              effluent_bod5
# DQO-S              effluent_cod
# SS-S               effluent_tss
#
# The original MVP also used:
#   dissolved_oxygen
#   temperature
#   hrt_hours
#
# These are not directly available in the selected UCI variables,
# so they are intentionally not fabricated or included here.
# ---------------------------------------------------------------------

SELECTED_COLUMNS = [
    "Q-E",
    "PH-E",
    "DBO-E",
    "DQO-E",
    "SS-E",
    "PH-S",
    "DBO-S",
    "DQO-S",
    "SS-S",
]


RENAME_MAP = {
    "Q-E": "flow_m3_day",
    "PH-E": "influent_ph",
    "DBO-E": "influent_bod5",
    "DQO-E": "influent_cod",
    "SS-E": "influent_tss",
    "PH-S": "effluent_ph",
    "DBO-S": "effluent_bod5",
    "DQO-S": "effluent_cod",
    "SS-S": "effluent_tss",
}


# ---------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------

def load_raw_dataset() -> pd.DataFrame:
    """Load the raw UCI Water Treatment Plant dataset."""

    if not RAW_FILE.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_FILE}\n"
            "Download water-treatment.data into data/raw/ first."
        )

    df = pd.read_csv(
        RAW_FILE,
        header=None,
        names=UCI_COLUMNS,
        na_values=["?", "", "NA", "N/A"],
        skipinitialspace=True,
    )

    return df


# ---------------------------------------------------------------------
# Data cleaning
# ---------------------------------------------------------------------

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Clean numeric values and prepare selected variables."""

    cleaned = df.copy()

    # Convert all UCI columns to numeric.
    #
    # Any unexpected non-numeric values become NaN instead of
    # silently remaining as strings.
    for column in cleaned.columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    # Keep only the variables required for the V2 model.
    cleaned = cleaned[SELECTED_COLUMNS].copy()

    # Rename UCI variables to application-friendly names.
    cleaned = cleaned.rename(columns=RENAME_MAP)

    # Remove rows where any selected feature or the target is missing.
    #
    # We do this rather than inventing measurements through
    # arbitrary imputation because this dataset is relatively small
    # and preserving real observations is preferable for the MVP.
    cleaned = cleaned.dropna(
        subset=[
            "flow_m3_day",
            "influent_ph",
            "influent_bod5",
            "influent_cod",
            "influent_tss",
            "effluent_ph",
            "effluent_bod5",
            "effluent_cod",
            "effluent_tss",
        ]
    ).copy()

    # Reset row numbering after removing incomplete observations.
    cleaned = cleaned.reset_index(drop=True)

    return cleaned


# ---------------------------------------------------------------------
# Basic validation
# ---------------------------------------------------------------------

def validate_dataset(df: pd.DataFrame) -> None:
    """Validate the processed dataset before saving."""

    expected_columns = [
        "flow_m3_day",
        "influent_ph",
        "influent_bod5",
        "influent_cod",
        "influent_tss",
        "effluent_ph",
        "effluent_bod5",
        "effluent_cod",
        "effluent_tss",
    ]

    if df.columns.tolist() != expected_columns:
        raise ValueError(
            "Processed dataset columns do not match expected columns."
        )

    if df.empty:
        raise ValueError(
            "Processed dataset contains zero rows."
        )

    if df.isna().any().any():
        raise ValueError(
            "Processed dataset still contains missing values."
        )

    if not df["flow_m3_day"].gt(0).all():
        raise ValueError(
            "flow_m3_day contains zero or negative values."
        )

    if not df["influent_bod5"].ge(0).all():
        raise ValueError(
            "influent_bod5 contains negative values."
        )

    if not df["influent_cod"].ge(0).all():
        raise ValueError(
            "influent_cod contains negative values."
        )

    if not df["influent_tss"].ge(0).all():
        raise ValueError(
            "influent_tss contains negative values."
        )

    if not df["effluent_bod5"].ge(0).all():
        raise ValueError(
            "effluent_bod5 contains negative values."
        )


# ---------------------------------------------------------------------
# Main processing pipeline
# ---------------------------------------------------------------------

def main() -> None:
    """Run the complete V2 dataset preparation pipeline."""

    print("=" * 70)
    print("Wastewater AI — V2 Dataset Preparation")
    print("=" * 70)

    print()
    print(f"Reading raw dataset:")
    print(f"  {RAW_FILE}")

    raw_df = load_raw_dataset()

    print()
    print("Raw dataset:")
    print(f"  Rows:    {len(raw_df)}")
    print(f"  Columns: {len(raw_df.columns)}")

    missing_before = raw_df.isna().sum().sum()

    print()
    print(f"Missing values before cleaning: {missing_before}")

    processed_df = clean_dataset(raw_df)

    print()
    print("Processed dataset:")
    print(f"  Rows:    {len(processed_df)}")
    print(f"  Columns: {len(processed_df.columns)}")

    print()
    print("Columns:")
    for column in processed_df.columns:
        print(f"  - {column}")

    validate_dataset(processed_df)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print("Validation: PASSED")

    print()
    print("Saved processed dataset:")
    print(f"  {OUTPUT_FILE}")

    print()
    print("First 5 processed rows:")
    print(
        processed_df.head().to_string(index=False)
    )

    print()
    print("Dataset summary:")
    print(
        processed_df.describe().to_string()
    )

    print()
    print("=" * 70)
    print("V2 dataset preparation complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()