from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "water-treatment.data"
)

OUTPUT_PATH = (
    BASE_DIR
    / "models"
    / "v2_feature_audit.csv"
)


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
    "RD-DBO-D",
    "RD-SS-D",
    "RD-DBO-G",
    "RD-SS-G",
    "RD-SED-G",
    "RD-N-NH4",
    "RD-N-NO2",
]


TARGET_COLUMN = "DBO-S"


def classify_feature(column):
    """
    Classify variables according to when/where they are measured.

    E = influent
    P/D/G = process/intermediate measurements
    S = effluent
    RD = removal-rate / derived process information
    """

    if column == TARGET_COLUMN:
        return "TARGET"

    if column.startswith("RD-"):
        return "DERIVED_PROCESS"

    if column.endswith("-E"):
        return "INFLUENT"

    if column.endswith("-P"):
        return "PROCESS"

    if column.endswith("-D"):
        return "PROCESS"

    if column.endswith("-G"):
        return "PROCESS"

    if column.endswith("-S"):
        return "EFFLUENT"

    return "UNKNOWN"


def leakage_risk(category):
    if category == "TARGET":
        return "TARGET"

    if category == "EFFLUENT":
        return "HIGH"

    if category == "DERIVED_PROCESS":
        return "MEDIUM"

    if category == "PROCESS":
        return "MEDIUM"

    if category == "INFLUENT":
        return "LOW"

    return "UNKNOWN"


def recommendation(category, column):
    if category == "TARGET":
        return "TARGET — never use as predictor"

    if category == "EFFLUENT":
        return "EXCLUDE — downstream leakage"

    if category == "INFLUENT":
        return "CANDIDATE — available before treatment"

    if category == "PROCESS":
        return (
            "INVESTIGATE — may be usable if measurement "
            "is available before prediction time"
        )

    if category == "DERIVED_PROCESS":
        return (
            "INVESTIGATE — verify whether calculation "
            "requires future/output data"
        )

    return "INVESTIGATE"


def main():
    print("=" * 70)
    print("WASTEWATER AI - V2.3 FULL UCI FEATURE AUDIT")
    print("=" * 70)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw UCI dataset not found: {RAW_DATA_PATH}"
        )

    # ---------------------------------------------------------
    # Load complete UCI dataset
    # ---------------------------------------------------------
    df = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=UCI_COLUMNS,
        na_values=["?", "", "NA", "N/A"],
    )

    # Convert every column to numeric.
    for column in UCI_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(f"\nRows: {len(df)}")
    print(f"Columns: {len(df.columns)}")

    # ---------------------------------------------------------
    # Build audit table
    # ---------------------------------------------------------
    audit_rows = []

    for column in UCI_COLUMNS:
        category = classify_feature(column)

        missing_count = int(
            df[column].isna().sum()
        )

        non_missing_count = int(
            df[column].notna().sum()
        )

        unique_count = int(
            df[column].nunique(dropna=True)
        )

        audit_rows.append(
            {
                "column": column,
                "category": category,
                "leakage_risk": leakage_risk(category),
                "missing_count": missing_count,
                "non_missing_count": non_missing_count,
                "unique_values": unique_count,
                "recommendation": recommendation(
                    category,
                    column,
                ),
            }
        )

    audit_df = pd.DataFrame(audit_rows)

    # ---------------------------------------------------------
    # Print complete audit
    # ---------------------------------------------------------
    print("\nComplete feature audit")
    print("-" * 70)

    print(
        audit_df.to_string(index=False)
    )

    # ---------------------------------------------------------
    # Category summary
    # ---------------------------------------------------------
    print("\nCategory summary")
    print("-" * 70)

    category_summary = (
        audit_df["category"]
        .value_counts()
        .rename_axis("category")
        .reset_index(name="count")
    )

    print(
        category_summary.to_string(index=False)
    )

    # ---------------------------------------------------------
    # Missing-data summary
    # ---------------------------------------------------------
    print("\nMissing-data summary")
    print("-" * 70)

    missing_df = audit_df[
        audit_df["missing_count"] > 0
    ].sort_values(
        by="missing_count",
        ascending=False,
    )

    if len(missing_df) == 0:
        print("No missing values found.")
    else:
        print(
            missing_df[
                [
                    "column",
                    "category",
                    "missing_count",
                    "non_missing_count",
                ]
            ].to_string(index=False)
        )

    # ---------------------------------------------------------
    # Candidate pre-treatment features
    # ---------------------------------------------------------
    candidate_features = audit_df[
        audit_df["category"] == "INFLUENT"
    ]["column"].tolist()

    print("\nInfluent candidate features")
    print("-" * 70)

    for feature in candidate_features:
        print(f"- {feature}")

    # ---------------------------------------------------------
    # Process variables for investigation
    # ---------------------------------------------------------
    process_features = audit_df[
        audit_df["category"].isin(
            ["PROCESS", "DERIVED_PROCESS"]
        )
    ]["column"].tolist()

    print("\nProcess / derived features requiring investigation")
    print("-" * 70)

    for feature in process_features:
        print(f"- {feature}")

    # ---------------------------------------------------------
    # Explicit leakage variables
    # ---------------------------------------------------------
    effluent_features = audit_df[
        audit_df["category"] == "EFFLUENT"
    ]["column"].tolist()

    print("\nEffluent variables — DO NOT USE AS PREDICTORS")
    print("-" * 70)

    for feature in effluent_features:
        print(f"- {feature}")

    # ---------------------------------------------------------
    # Correlation with target
    # ---------------------------------------------------------
    numeric_df = df[UCI_COLUMNS]

    correlations = (
        numeric_df.corr(numeric_only=True)[TARGET_COLUMN]
        .drop(TARGET_COLUMN)
        .sort_values(
            key=lambda series: series.abs(),
            ascending=False,
        )
    )

    print("\nCorrelation with target DBO-S")
    print("-" * 70)

    for column, correlation in correlations.items():
        print(
            f"{column:12s} "
            f"{correlation: .4f}"
        )

    # ---------------------------------------------------------
    # Save audit
    # ---------------------------------------------------------
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    audit_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nAudit saved:")
    print(OUTPUT_PATH)

    print("\nFeature audit completed successfully.")


if __name__ == "__main__":
    main()