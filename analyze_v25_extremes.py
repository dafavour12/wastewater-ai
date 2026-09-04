from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent

RAW_DATA = BASE_DIR / "data" / "raw" / "water-treatment.data"
ERROR_FILE = BASE_DIR / "models" / "v24_random_predictions.csv"
CHRONO_FILE = BASE_DIR / "models" / "v24_chronological_predictions.csv"

OUTLIER_FILE = BASE_DIR / "models" / "v25_extreme_observations.csv"
SUMMARY_FILE = BASE_DIR / "models" / "v25_extreme_case_summary.txt"


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


FEATURE_MAP = {
    "Q-E": "flow_m3_day",
    "ZN-E": "influent_zinc",
    "PH-E": "influent_ph",
    "DBO-E": "influent_bod5",
    "DQO-E": "influent_cod",
    "SS-E": "influent_tss",
    "SSV-E": "influent_vss",
    "SED-E": "influent_sediments",
    "COND-E": "influent_conductivity",
    "DBO-S": "effluent_bod5",
}


def load_raw_data():
    df = pd.read_csv(
        RAW_DATA,
        header=None,
        names=UCI_COLUMNS,
        na_values=["?", ""],
    )

    for column in UCI_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def load_predictions(path):
    df = pd.read_csv(path)

    required = {
        "actual_effluent_bod5",
        "predicted_effluent_bod5",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            f"{path.name} is missing required columns: {sorted(missing)}"
        )

    return df


def print_distribution(df):
    target = df["DBO-S"].dropna()

    print("\n" + "-" * 70)
    print("EFFLUENT BOD5 DISTRIBUTION")
    print("-" * 70)

    print(f"Observations: {len(target)}")
    print(f"Minimum: {target.min():.2f} mg/L")
    print(f"25th percentile: {target.quantile(0.25):.2f} mg/L")
    print(f"Median: {target.median():.2f} mg/L")
    print(f"Mean: {target.mean():.2f} mg/L")
    print(f"75th percentile: {target.quantile(0.75):.2f} mg/L")
    print(f"90th percentile: {target.quantile(0.90):.2f} mg/L")
    print(f"95th percentile: {target.quantile(0.95):.2f} mg/L")
    print(f"99th percentile: {target.quantile(0.99):.2f} mg/L")
    print(f"Maximum: {target.max():.2f} mg/L")


def identify_extremes(df):
    target = df["DBO-S"]

    thresholds = {
        "high_30_plus": target >= 30,
        "very_high_50_plus": target >= 50,
        "extreme_75_plus": target >= 75,
        "extreme_100_plus": target >= 100,
    }

    print("\n" + "-" * 70)
    print("EXTREME CASE COUNTS")
    print("-" * 70)

    for name, mask in thresholds.items():
        print(f"{name}: {int(mask.sum())}")

    extreme_mask = target >= 50

    return df.loc[extreme_mask].copy()


def analyze_feature_values(extreme_df):
    print("\n" + "-" * 70)
    print("EXTREME-CASE FEATURE INSPECTION")
    print("-" * 70)

    selected = [
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
    ]

    available = [column for column in selected if column in extreme_df.columns]

    if extreme_df.empty:
        print("No observations with effluent BOD5 >= 50 mg/L.")
        return

    display_df = extreme_df[available].copy()

    with pd.option_context(
        "display.max_columns",
        None,
        "display.width",
        240,
        "display.max_rows",
        100,
    ):
        print(display_df.to_string(index=True))


def calculate_feature_zscores(df):
    numeric = df[UCI_COLUMNS].copy()

    means = numeric.mean()
    stds = numeric.std(ddof=0).replace(0, np.nan)

    zscores = (numeric - means) / stds

    return zscores


def analyze_152_case(df):
    target = df["DBO-S"]

    candidates = df.loc[target >= 100].copy()

    print("\n" + "-" * 70)
    print(">=100 MG/L CASE INVESTIGATION")
    print("-" * 70)

    if candidates.empty:
        print("No observations with effluent BOD5 >= 100 mg/L.")
        return candidates

    zscores = calculate_feature_zscores(df)

    for index, row in candidates.iterrows():
        print(f"\nObservation index: {index}")
        print(f"Effluent BOD5: {row['DBO-S']:.2f} mg/L")

        print("\nRaw influent/process values:")

        for column in [
            "Q-E",
            "ZN-E",
            "PH-E",
            "DBO-E",
            "DQO-E",
            "SS-E",
            "SSV-E",
            "SED-E",
            "COND-E",
        ]:
            value = row[column]

            if pd.isna(value):
                print(f"  {column}: missing")
            else:
                print(f"  {column}: {value:.3f}")

        print("\nFeature z-scores:")

        row_zscores = zscores.loc[index].dropna()

        unusual = row_zscores.abs().sort_values(ascending=False)

        for column, value in unusual.head(10).items():
            print(f"  {column}: z={value:.2f}")


def compare_extreme_predictions(raw_df, random_predictions, chronological):
    print("\n" + "-" * 70)
    print("PREDICTION COVERAGE OF EXTREME CASES")
    print("-" * 70)

    extreme_actuals = raw_df.loc[raw_df["DBO-S"] >= 50, "DBO-S"]

    print(
        f"Raw observations >=50 mg/L: "
        f"{len(extreme_actuals)}"
    )

    if extreme_actuals.empty:
        print("No extreme observations available.")
        return

    print(
        "Note: prediction files contain holdout observations only, "
        "so not every raw extreme observation necessarily appears there."
    )

    for name, predictions in [
        ("Random holdout", random_predictions),
        ("Chronological holdout", chronological),
    ]:
        extreme_predictions = predictions[
            predictions["actual_effluent_bod5"] >= 50
        ].copy()

        print(f"\n{name}:")
        print(
            f"Extreme holdout observations: "
            f"{len(extreme_predictions)}"
        )

        if not extreme_predictions.empty:
            extreme_predictions["error"] = (
                extreme_predictions["actual_effluent_bod5"]
                - extreme_predictions["predicted_effluent_bod5"]
            )

            with pd.option_context(
                "display.width",
                180,
                "display.max_columns",
                20,
            ):
                print(
                    extreme_predictions[
                        [
                            "actual_effluent_bod5",
                            "predicted_effluent_bod5",
                            "error",
                        ]
                    ].to_string(index=False)
                )


def build_extreme_output(raw_df):
    output = raw_df.copy()

    output["target_extreme_30_plus"] = output["DBO-S"] >= 30
    output["target_extreme_50_plus"] = output["DBO-S"] >= 50
    output["target_extreme_75_plus"] = output["DBO-S"] >= 75
    output["target_extreme_100_plus"] = output["DBO-S"] >= 100

    return output


def write_summary(df, extreme_df):
    target = df["DBO-S"].dropna()

    lines = [
        "WASTEWATER AI - V2.5 EXTREME-CASE INVESTIGATION",
        "=" * 60,
        "",
        f"Raw observations: {len(df)}",
        f"Valid target observations: {len(target)}",
        "",
        "EFFLUENT BOD5 DISTRIBUTION",
        "-" * 40,
        f"Minimum: {target.min():.4f}",
        f"Median: {target.median():.4f}",
        f"Mean: {target.mean():.4f}",
        f"90th percentile: {target.quantile(0.90):.4f}",
        f"95th percentile: {target.quantile(0.95):.4f}",
        f"99th percentile: {target.quantile(0.99):.4f}",
        f"Maximum: {target.max():.4f}",
        "",
        "EXTREME COUNTS",
        "-" * 40,
        f">=30 mg/L: {(target >= 30).sum()}",
        f">=50 mg/L: {(target >= 50).sum()}",
        f">=75 mg/L: {(target >= 75).sum()}",
        f">=100 mg/L: {(target >= 100).sum()}",
        "",
        "EXTREME OBSERVATIONS",
        "-" * 40,
        f"Rows with BOD5 >=50 mg/L: {len(extreme_df)}",
    ]

    if not extreme_df.empty:
        lines.append("")
        lines.append("Extreme target values:")
        lines.extend(
            f"  index {index}: {row['DBO-S']:.4f} mg/L"
            for index, row in extreme_df.iterrows()
        )

    SUMMARY_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    print("=" * 70)
    print("WASTEWATER AI - V2.5 EXTREME-CASE & DATA-QUALITY INVESTIGATION")
    print("=" * 70)

    raw_df = load_raw_data()

    random_predictions = load_predictions(ERROR_FILE)
    chronological_predictions = load_predictions(CHRONO_FILE)

    print(f"\nRaw dataset loaded: {len(raw_df)} observations")

    print_distribution(raw_df)

    extreme_df = identify_extremes(raw_df)

    analyze_feature_values(extreme_df)

    analyze_152_case(raw_df)

    compare_extreme_predictions(
        raw_df,
        random_predictions,
        chronological_predictions,
    )

    output = build_extreme_output(raw_df)

    output.to_csv(
        OUTLIER_FILE,
        index=True,
    )

    write_summary(
        raw_df,
        extreme_df,
    )

    print("\n" + "=" * 70)
    print("V2.5 INVESTIGATION FILES SAVED")
    print("=" * 70)

    print(OUTLIER_FILE)
    print(SUMMARY_FILE)

    print("\n" + "=" * 70)
    print("V2.5 EXTREME-CASE INVESTIGATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()