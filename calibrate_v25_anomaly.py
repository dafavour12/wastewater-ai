from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_PATH = PROJECT_ROOT / "data" / "raw" / "water-treatment.data"
OUTPUT_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


UCI_COLUMNS = [
    "date",
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


PROCESS_COLUMNS = [
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


def load_dataset():
    """Load and numerically convert the raw UCI dataset."""
    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=UCI_COLUMNS,
        na_values=["?", "", "NA", "NaN"],
    )

    for column in UCI_COLUMNS[1:]:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


def calculate_metrics(y_true, y_pred):
    """Calculate binary alarm-classification metrics."""
    false_positive_rate = (
        ((y_pred == 1) & (y_true == 0)).sum()
        / max((y_true == 0).sum(), 1)
    )

    return {
        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0,
        ),
        "false_positive_rate": false_positive_rate,
        "predicted_anomalies": int(y_pred.sum()),
    }


def add_risk_band(percentile):
    """Convert anomaly percentile into a candidate risk band."""
    if percentile < 90:
        return "NORMAL"

    if percentile < 97:
        return "LOW"

    if percentile < 99:
        return "ELEVATED"

    if percentile < 99.5:
        return "HIGH"

    return "CRITICAL"


def main():
    print("=" * 72)
    print("V2.5.1 ANOMALY CALIBRATION")
    print("=" * 72)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------

    df = load_dataset()

    raw_count = len(df)

    valid_target = df[TARGET_COLUMN].notna()

    df = df.loc[valid_target].copy()

    valid_count = len(df)

    print(f"Raw dataset observations: {raw_count}")
    print(f"Valid target observations: {valid_count}")
    print(
        "Removed target-missing observations: "
        f"{raw_count - valid_count}"
    )
    print()

    # ------------------------------------------------------------------
    # Process-variable anomaly detector
    # ------------------------------------------------------------------

    X = df[PROCESS_COLUMNS].copy()

    y = df[TARGET_COLUMN].astype(float)

    detector = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                RobustScaler(),
            ),
            (
                "model",
                IsolationForest(
                    n_estimators=1000,
                    contamination="auto",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    detector.fit(X)

    model = detector.named_steps["model"]

    X_imputed = detector.named_steps["imputer"].transform(X)

    X_scaled = detector.named_steps["scaler"].transform(
        X_imputed
    )

    # Isolation Forest score_samples:
    # higher = more normal.
    #
    # We negate it so that:
    # higher anomaly_score = more anomalous.
    raw_scores = -model.score_samples(X_scaled)

    df["anomaly_score"] = raw_scores

    # Higher percentile = more anomalous.
    df["anomaly_percentile"] = (
        pd.Series(
            raw_scores,
            index=df.index,
        )
        .rank(
            method="average",
            pct=True,
        )
        * 100
    )

    df["risk_band"] = df[
        "anomaly_percentile"
    ].apply(add_risk_band)

    # ------------------------------------------------------------------
    # Candidate risk-band distribution
    # ------------------------------------------------------------------

    print("CANDIDATE RISK-BAND DISTRIBUTION")

    band_order = [
        "NORMAL",
        "LOW",
        "ELEVATED",
        "HIGH",
        "CRITICAL",
    ]

    band_counts = df["risk_band"].value_counts()

    for band in band_order:
        count = int(
            band_counts.get(
                band,
                0,
            )
        )

        percentage = (
            count / valid_count * 100
        )

        print(
            f"{band:<10}: "
            f"{count:>3} "
            f"({percentage:.1f}%)"
        )

    print()

    # ------------------------------------------------------------------
    # Alarm-budget calibration
    # ------------------------------------------------------------------

    alarm_budgets = [
        0.5,
        1.0,
        2.0,
        3.0,
        5.0,
    ]

    target_thresholds = [
        30,
        50,
        75,
    ]

    calibration_rows = []

    print("ALARM-BUDGET CALIBRATION")
    print()

    for budget in alarm_budgets:
        cutoff = np.percentile(
            raw_scores,
            100 - budget,
        )

        alerts = raw_scores >= cutoff

        print(
            f"Top {budget:.1f}%: "
            f"Alerts {int(alerts.sum())}"
        )

        for target_threshold in target_thresholds:
            y_true = (
                y >= target_threshold
            ).astype(int)

            y_pred = alerts.astype(int)

            metrics = calculate_metrics(
                y_true,
                y_pred,
            )

            print(
                f"  BOD5>={target_threshold:<2} "
                f"precision="
                f"{metrics['precision']:.3f} "
                f"recall="
                f"{metrics['recall']:.3f} "
                f"F1="
                f"{metrics['f1']:.3f} "
                f"FPR="
                f"{metrics['false_positive_rate']:.3f}"
            )

            calibration_rows.append(
                {
                    "alarm_budget_percent": budget,
                    "alert_count": int(
                        alerts.sum()
                    ),
                    "target_threshold": (
                        target_threshold
                    ),
                    **metrics,
                }
            )

        print()

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    calibration_path = (
        OUTPUT_DIR
        / "v25_anomaly_calibration.csv"
    )

    calibration_df.to_csv(
        calibration_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Explicit extreme-case audit
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    # Rename DBO-S to effluent_bod5 here so all downstream
    # reporting uses the same human-readable target name.
    # ------------------------------------------------------------------

    extreme_mask = y >= 50

    extreme_cases = df.loc[
        extreme_mask,
        [
            "date",
            TARGET_COLUMN,
            "anomaly_score",
            "anomaly_percentile",
            "risk_band",
        ],
    ].copy()

    extreme_cases = extreme_cases.rename(
        columns={
            TARGET_COLUMN: "effluent_bod5",
        }
    )

    extreme_cases = extreme_cases.sort_values(
        "effluent_bod5",
        ascending=False,
    )

    # ------------------------------------------------------------------
    # Mark which extreme cases are caught at each alarm budget.
    # ------------------------------------------------------------------

    for budget in alarm_budgets:
        cutoff = np.percentile(
            raw_scores,
            100 - budget,
        )

        column_name = (
            f"top_{str(budget).replace('.', '_')}pct"
        )

        extreme_cases[column_name] = (
            extreme_cases["anomaly_score"]
            >= cutoff
        )

    # Candidate risk-band flags.

    extreme_cases["elevated_or_higher"] = (
        extreme_cases["anomaly_percentile"]
        >= 97
    )

    extreme_cases["high_or_higher"] = (
        extreme_cases["anomaly_percentile"]
        >= 99
    )

    extreme_cases["critical"] = (
        extreme_cases["anomaly_percentile"]
        >= 99.5
    )

    extreme_cases_path = (
        OUTPUT_DIR
        / "v25_anomaly_extreme_cases.csv"
    )

    extreme_cases.to_csv(
        extreme_cases_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Extreme-case console audit
    # ------------------------------------------------------------------

    print("=" * 72)
    print("EXTREME-CASE AUDIT")
    print("=" * 72)

    print(
        "Extreme observations "
        f"(BOD5 >= 50): {len(extreme_cases)}"
    )

    print()

    display_columns = [
        "date",
        "effluent_bod5",
        "anomaly_percentile",
        "risk_band",
        "top_0_5pct",
        "top_1_0pct",
        "top_2_0pct",
        "top_3_0pct",
        "top_5_0pct",
    ]

    print(
        extreme_cases[
            display_columns
        ].to_string(index=False)
    )

    print()

    # ------------------------------------------------------------------
    # Extreme-event capture summary
    # ------------------------------------------------------------------

    print("EXTREME-EVENT CAPTURE SUMMARY")
    print()

    for budget in alarm_budgets:
        column_name = (
            f"top_{str(budget).replace('.', '_')}pct"
        )

        caught = int(
            extreme_cases[column_name].sum()
        )

        total = len(extreme_cases)

        recall = (
            caught / total
            if total > 0
            else 0
        )

        print(
            f"Top {budget:.1f}% alarm budget: "
            f"{caught}/{total} extreme events "
            f"caught "
            f"({recall:.1%})"
        )

    print()

    # ------------------------------------------------------------------
    # Candidate risk-band evaluation
    # ------------------------------------------------------------------

    band_thresholds = {
        "NORMAL": 0,
        "LOW": 90,
        "ELEVATED": 97,
        "HIGH": 99,
        "CRITICAL": 99.5,
    }

    print(
        "CANDIDATE RISK-BAND EVALUATION"
    )

    print()

    band_rows = []

    for band, percentile_threshold in (
        band_thresholds.items()
    ):
        alerts = (
            df["anomaly_percentile"]
            >= percentile_threshold
        )

        y_true = (
            y >= 50
        ).astype(int)

        y_pred = alerts.astype(int)

        metrics = calculate_metrics(
            y_true,
            y_pred,
        )

        print(
            f"Alarm at {band:<9} "
            f"or higher: "
            f"{int(alerts.sum())} alerts, "
            f"precision="
            f"{metrics['precision']:.3f}, "
            f"recall="
            f"{metrics['recall']:.3f}, "
            f"F1="
            f"{metrics['f1']:.3f}, "
            f"FPR="
            f"{metrics['false_positive_rate']:.3f}"
        )

        band_rows.append(
            {
                "risk_band_threshold": band,
                "percentile_threshold": (
                    percentile_threshold
                ),
                "target": "BOD5 >= 50",
                **metrics,
            }
        )

    print()

    band_df = pd.DataFrame(
        band_rows
    )

    # ------------------------------------------------------------------
    # Ranking output
    # ------------------------------------------------------------------

    rankings = df[
        [
            "date",
            TARGET_COLUMN,
            "anomaly_score",
            "anomaly_percentile",
            "risk_band",
        ]
    ].copy()

    rankings = rankings.rename(
        columns={
            TARGET_COLUMN: "effluent_bod5",
        }
    )

    rankings = rankings.sort_values(
        "anomaly_score",
        ascending=False,
    )

    rankings_path = (
        OUTPUT_DIR
        / "v25_anomaly_rankings.csv"
    )

    rankings.to_csv(
        rankings_path,
        index=False,
    )

    # ------------------------------------------------------------------
    # Summary values
    # ------------------------------------------------------------------

    top_0_5_cutoff = np.percentile(
        raw_scores,
        99.5,
    )

    top_1_cutoff = np.percentile(
        raw_scores,
        99.0,
    )

    top_2_cutoff = np.percentile(
        raw_scores,
        98.0,
    )

    top_3_cutoff = np.percentile(
        raw_scores,
        97.0,
    )

    top_5_cutoff = np.percentile(
        raw_scores,
        95.0,
    )

    summary_path = (
        OUTPUT_DIR
        / "v25_anomaly_calibration_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            "WASTEWATER AI V2.5.1 "
            "ANOMALY CALIBRATION\n"
        )

        f.write("=" * 60 + "\n\n")

        f.write(
            f"Raw dataset observations: "
            f"{raw_count}\n"
        )

        f.write(
            f"Valid target observations: "
            f"{valid_count}\n"
        )

        f.write(
            "Removed target-missing observations: "
            f"{raw_count - valid_count}\n\n"
        )

        f.write(
            "Detector: Isolation Forest\n"
        )

        f.write(
            "Trees: 1000\n"
        )

        f.write(
            "Input: 22 process variables\n"
        )

        f.write(
            "Effluent variables excluded "
            "from detector input.\n\n"
        )

        f.write(
            "Candidate risk bands:\n"
        )

        f.write(
            "NORMAL: 0-90 percentile\n"
        )

        f.write(
            "LOW: 90-97 percentile\n"
        )

        f.write(
            "ELEVATED: 97-99 percentile\n"
        )

        f.write(
            "HIGH: 99-99.5 percentile\n"
        )

        f.write(
            "CRITICAL: 99.5-100 percentile\n\n"
        )

        f.write(
            "Alarm budget thresholds:\n"
        )

        f.write(
            f"Top 0.5% cutoff score: "
            f"{top_0_5_cutoff:.6f}\n"
        )

        f.write(
            f"Top 1% cutoff score: "
            f"{top_1_cutoff:.6f}\n"
        )

        f.write(
            f"Top 2% cutoff score: "
            f"{top_2_cutoff:.6f}\n"
        )

        f.write(
            f"Top 3% cutoff score: "
            f"{top_3_cutoff:.6f}\n"
        )

        f.write(
            f"Top 5% cutoff score: "
            f"{top_5_cutoff:.6f}\n\n"
        )

        f.write(
            "Extreme-event capture:\n"
        )

        for budget in alarm_budgets:
            column_name = (
                f"top_{str(budget).replace('.', '_')}pct"
            )

            caught = int(
                extreme_cases[
                    column_name
                ].sum()
            )

            total = len(
                extreme_cases
            )

            recall = (
                caught / total
                if total > 0
                else 0
            )

            f.write(
                f"Top {budget:.1f}%: "
                f"{caught}/{total} "
                f"({recall:.1%})\n"
            )

        f.write("\n")

        f.write(
            "Extreme-case details:\n"
        )

        for _, row in (
            extreme_cases.iterrows()
        ):
            f.write(
                f"{row['date']}: "
                f"BOD5="
                f"{row['effluent_bod5']:.1f}, "
                f"percentile="
                f"{row['anomaly_percentile']:.3f}, "
                f"risk="
                f"{row['risk_band']}\n"
            )

        f.write("\n")

        f.write(
            "Interpretation:\n"
        )

        f.write(
            "- Isolation Forest provides a useful "
            "research anomaly ranking.\n"
        )

        f.write(
            "- Extreme BOD5 events are generally "
            "ranked as high-anomaly observations.\n"
        )

        f.write(
            "- Alarm thresholds remain research "
            "candidates because only six "
            "BOD5>=50 observations exist.\n"
        )

        f.write(
            "- The detector must not be treated "
            "as production validated without "
            "additional operational data.\n"
        )

    print("OUTPUT FILES")
    print("-" * 72)

    print(calibration_path)
    print(extreme_cases_path)
    print(rankings_path)
    print(summary_path)

    print()

    print("=" * 72)
    print("V2.5.1 CALIBRATION COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()
