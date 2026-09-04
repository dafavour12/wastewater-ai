from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# ======================================================================
# CONFIGURATION
# ======================================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "data" / "raw" / "water-treatment.data"
MODELS_DIR = BASE_DIR / "models"

COMPARISON_OUTPUT = MODELS_DIR / "v25_process_anomaly_comparison.csv"
PREDICTIONS_OUTPUT = MODELS_DIR / "v25_process_anomaly_predictions.csv"
SUMMARY_OUTPUT = MODELS_DIR / "v25_process_anomaly_summary.txt"


# ======================================================================
# UCI DATASET COLUMNS
# ======================================================================

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


# ======================================================================
# PROCESS VARIABLES
# ======================================================================

PROCESS_FEATURES = [
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


# ======================================================================
# DATA LOADING
# ======================================================================

def load_data() -> pd.DataFrame:
    """Load and numerically coerce the UCI wastewater dataset."""

    df = pd.read_csv(
        RAW_DATA,
        header=None,
        names=UCI_COLUMNS,
        na_values=["?", ""],
        skipinitialspace=True,
    )

    for column in UCI_COLUMNS[1:]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["effluent_bod5"] = df["DBO-S"]

    return df


# ======================================================================
# TARGET LABELS
# ======================================================================

def add_target_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Create offline evaluation labels from effluent BOD5."""

    target = df["effluent_bod5"]

    df["elevated_30"] = (
        target >= 30
    ).astype(int)

    df["extreme_50"] = (
        target >= 50
    ).astype(int)

    df["severe_75"] = (
        target >= 75
    ).astype(int)

    return df


# ======================================================================
# ROBUST Z-SCORE DETECTOR
# ======================================================================

def robust_zscore_scores(
    X: pd.DataFrame,
) -> np.ndarray:
    """
    Calculate a row-level anomaly score using robust z-scores.

    Median and MAD are used instead of mean and standard deviation,
    making the detector less sensitive to the extreme observations
    being investigated.
    """

    X_imputed = SimpleImputer(strategy="median").fit_transform(X)

    X_df = pd.DataFrame(
        X_imputed,
        columns=X.columns,
        index=X.index,
    )

    medians = X_df.median(axis=0)

    mad = (
        X_df.sub(medians)
        .abs()
        .median(axis=0)
    )

    mad = mad.replace(0, np.nan)

    robust_z = (
        0.6745
        * X_df.sub(medians)
        / mad
    )

    robust_z = robust_z.replace(
        [np.inf, -np.inf],
        np.nan,
    ).fillna(0.0)

    scores = robust_z.abs().max(axis=1)

    return scores.to_numpy()


# ======================================================================
# ISOLATION FOREST
# ======================================================================

def isolation_forest_scores(
    X: pd.DataFrame,
    contamination: float,
) -> np.ndarray:
    """
    Fit Isolation Forest and return anomaly scores.

    Higher returned values indicate greater anomaly likelihood.
    """

    pipeline = Pipeline(
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
                    n_estimators=500,
                    contamination=contamination,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    pipeline.fit(X)

    model = pipeline.named_steps["model"]

    raw_scores = -model.score_samples(
        pipeline.named_steps["scaler"].transform(
            pipeline.named_steps["imputer"].transform(X)
        )
    )

    return raw_scores


# ======================================================================
# METRICS
# ======================================================================

def calculate_metrics(
    actual: pd.Series,
    predicted_anomaly: np.ndarray,
    threshold_name: str,
    detector_name: str,
    feature_set: str,
    anomaly_threshold: float,
) -> dict:

    actual = actual.astype(int)
    predicted_anomaly = predicted_anomaly.astype(int)

    cm = confusion_matrix(
        actual,
        predicted_anomaly,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(
        actual,
        predicted_anomaly,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        predicted_anomaly,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        predicted_anomaly,
        zero_division=0,
    )

    false_positive_rate = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0.0
    )

    anomaly_rate = predicted_anomaly.mean()

    return {
        "feature_set": feature_set,
        "detector": detector_name,
        "target_threshold": threshold_name,
        "anomaly_threshold": anomaly_threshold,
        "observations": len(actual),
        "actual_events": int(actual.sum()),
        "predicted_anomalies": int(predicted_anomaly.sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "anomaly_rate": anomaly_rate,
    }


# ======================================================================
# DETECTOR EVALUATION
# ======================================================================

def evaluate_detector(
    scores: np.ndarray,
    df: pd.DataFrame,
    detector_name: str,
    feature_set: str,
    threshold_name: str,
    target_column: str,
    thresholds: list[float],
) -> list[dict]:

    results = []

    actual = df[target_column]

    for threshold in thresholds:

        predicted = (
            scores >= threshold
        ).astype(int)

        results.append(
            calculate_metrics(
                actual=actual,
                predicted_anomaly=predicted,
                threshold_name=threshold_name,
                detector_name=detector_name,
                feature_set=feature_set,
                anomaly_threshold=threshold,
            )
        )

    return results


# ======================================================================
# MAIN ANALYSIS
# ======================================================================

def main() -> None:

    print("=" * 70)
    print("WASTEWATER AI - V2.5 PROCESS ANOMALY DETECTION")
    print("=" * 70)

    MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = load_data()
    df = add_target_labels(df)

    print()
    print(f"Raw dataset observations: {len(df)}")

    valid_target = df[
        "effluent_bod5"
    ].notna()

    df_valid = df.loc[
        valid_target
    ].copy()

    print(
        f"Valid target observations: {len(df_valid)}"
    )

    print()
    print("-" * 70)
    print("OFFLINE EVENT DISTRIBUTION")
    print("-" * 70)

    for label, column in [
        ("BOD5 >= 30", "elevated_30"),
        ("BOD5 >= 50", "extreme_50"),
        ("BOD5 >= 75", "severe_75"),
    ]:
        count = int(
            df_valid[column].sum()
        )

        percentage = (
            100 * count / len(df_valid)
        )

        print(
            f"{label:<12}: "
            f"{count:3d} "
            f"({percentage:5.1f}%)"
        )

    X = df_valid[
        PROCESS_FEATURES
    ].copy()

    print()
    print("-" * 70)
    print("PROCESS FEATURE SET")
    print("-" * 70)

    print(
        f"Process variables: {len(PROCESS_FEATURES)}"
    )

    for feature in PROCESS_FEATURES:
        print(f"  {feature}")

    # --------------------------------------------------------------
    # Detector 1: robust z-score
    # --------------------------------------------------------------

    print()
    print("-" * 70)
    print("ROBUST Z-SCORE ANALYSIS")
    print("-" * 70)

    robust_scores = robust_zscore_scores(X)

    print(
        f"Maximum robust anomaly score: "
        f"{robust_scores.max():.2f}"
    )

    # Evaluate several operational thresholds.
    robust_thresholds = [
        2.5,
        3.0,
        3.5,
        4.0,
        5.0,
    ]

    results = []

    results.extend(
        evaluate_detector(
            scores=robust_scores,
            df=df_valid,
            detector_name="Robust Z-Score",
            feature_set="process",
            threshold_name="BOD5 >= 30",
            target_column="elevated_30",
            thresholds=robust_thresholds,
        )
    )

    results.extend(
        evaluate_detector(
            scores=robust_scores,
            df=df_valid,
            detector_name="Robust Z-Score",
            feature_set="process",
            threshold_name="BOD5 >= 50",
            target_column="extreme_50",
            thresholds=robust_thresholds,
        )
    )

    results.extend(
        evaluate_detector(
            scores=robust_scores,
            df=df_valid,
            detector_name="Robust Z-Score",
            feature_set="process",
            threshold_name="BOD5 >= 75",
            target_column="severe_75",
            thresholds=robust_thresholds,
        )
    )

    # --------------------------------------------------------------
    # Detector 2: Isolation Forest
    # --------------------------------------------------------------

    print()
    print("-" * 70)
    print("ISOLATION FOREST ANALYSIS")
    print("-" * 70)

    isolation_configs = [
        ("Isolation Forest 1%", 0.01),
        ("Isolation Forest 2%", 0.02),
        ("Isolation Forest 5%", 0.05),
        ("Isolation Forest 10%", 0.10),
    ]

    prediction_records = pd.DataFrame(
        {
            "date": df_valid["date"].values,
            "effluent_bod5": df_valid[
                "effluent_bod5"
            ].values,
            "elevated_30": df_valid[
                "elevated_30"
            ].values,
            "extreme_50": df_valid[
                "extreme_50"
            ].values,
            "severe_75": df_valid[
                "severe_75"
            ].values,
            "robust_z_score": robust_scores,
        }
    )

    for detector_name, contamination in isolation_configs:

        scores = isolation_forest_scores(
            X,
            contamination=contamination,
        )

        prediction_records[
            detector_name.lower()
            .replace(" ", "_")
            .replace("%", "pct")
        ] = scores

        # Isolation Forest's native decision threshold
        # corresponds to the contamination setting.
        anomaly_count = max(
            1,
            int(
                np.ceil(
                    contamination
                    * len(df_valid)
                )
            ),
        )

        cutoff = np.sort(scores)[
            -anomaly_count
        ]

        results.extend(
            evaluate_detector(
                scores=scores,
                df=df_valid,
                detector_name=detector_name,
                feature_set="process",
                threshold_name="BOD5 >= 30",
                target_column="elevated_30",
                thresholds=[cutoff],
            )
        )

        results.extend(
            evaluate_detector(
                scores=scores,
                df=df_valid,
                detector_name=detector_name,
                feature_set="process",
                threshold_name="BOD5 >= 50",
                target_column="extreme_50",
                thresholds=[cutoff],
            )
        )

        results.extend(
            evaluate_detector(
                scores=scores,
                df=df_valid,
                detector_name=detector_name,
                feature_set="process",
                threshold_name="BOD5 >= 75",
                target_column="severe_75",
                thresholds=[cutoff],
            )
        )

    # --------------------------------------------------------------
    # Save results
    # --------------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        [
            "target_threshold",
            "f1",
            "recall",
        ],
        ascending=[
            True,
            False,
            False,
        ],
    )

    results_df.to_csv(
        COMPARISON_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------------
    # Best detector predictions
    # --------------------------------------------------------------

    robust_best_threshold = 3.5

    prediction_records[
        "robust_anomaly"
    ] = (
        prediction_records[
            "robust_z_score"
        ]
        >= robust_best_threshold
    ).astype(int)

    print()
    print("-" * 70)
    print("ROBUST Z-SCORE @ 3.5")
    print("-" * 70)

    for label, column in [
        ("BOD5 >= 30", "elevated_30"),
        ("BOD5 >= 50", "extreme_50"),
        ("BOD5 >= 75", "severe_75"),
    ]:

        actual = prediction_records[
            column
        ].astype(int)

        predicted = prediction_records[
            "robust_anomaly"
        ].astype(int)

        precision = precision_score(
            actual,
            predicted,
            zero_division=0,
        )

        recall = recall_score(
            actual,
            predicted,
            zero_division=0,
        )

        f1 = f1_score(
            actual,
            predicted,
            zero_division=0,
        )

        print()
        print(label)
        print(
            f"  Precision: {precision:.3f}"
        )
        print(
            f"  Recall:    {recall:.3f}"
        )
        print(
            f"  F1:        {f1:.3f}"
        )

    prediction_records.to_csv(
        PREDICTIONS_OUTPUT,
        index=False,
    )

    # --------------------------------------------------------------
    # Extreme cases
    # --------------------------------------------------------------

    print()
    print("-" * 70)
    print("EXTREME CASE DETECTION")
    print("-" * 70)

    extreme_cases = prediction_records[
        prediction_records[
            "extreme_50"
        ] == 1
    ].copy()

    print(
        f"Extreme observations: "
        f"{len(extreme_cases)}"
    )

    print()

    display_columns = [
        "date",
        "effluent_bod5",
        "robust_z_score",
        "robust_anomaly",
    ]

    print(
        extreme_cases[
            display_columns
        ].to_string(index=False)
    )

    # --------------------------------------------------------------
    # Summary
    # --------------------------------------------------------------

    best_extreme = results_df[
        results_df[
            "target_threshold"
        ] == "BOD5 >= 50"
    ].sort_values(
        ["f1", "recall"],
        ascending=False,
    ).head(1)

    best_severe = results_df[
        results_df[
            "target_threshold"
        ] == "BOD5 >= 75"
    ].sort_values(
        ["f1", "recall"],
        ascending=False,
    ).head(1)

    summary_lines = [
        "WASTEWATER AI - V2.5 PROCESS ANOMALY DETECTION",
        "=" * 60,
        "",
        f"Raw observations: {len(df)}",
        f"Valid target observations: {len(df_valid)}",
        f"Process variables: {len(PROCESS_FEATURES)}",
        "",
        "Offline event counts:",
        f"  BOD5 >= 30: {int(df_valid['elevated_30'].sum())}",
        f"  BOD5 >= 50: {int(df_valid['extreme_50'].sum())}",
        f"  BOD5 >= 75: {int(df_valid['severe_75'].sum())}",
        "",
        "Important methodology:",
        "  Process variables only were used as anomaly inputs.",
        "  Effluent variables were excluded from detector inputs.",
        "  Effluent BOD5 was used only as an offline evaluation target.",
        "",
    ]

    if not best_extreme.empty:
        row = best_extreme.iloc[0]

        summary_lines.extend(
            [
                "Best BOD5 >= 50 detector:",
                f"  Detector: {row['detector']}",
                f"  Precision: {row['precision']:.3f}",
                f"  Recall: {row['recall']:.3f}",
                f"  F1: {row['f1']:.3f}",
                f"  False-positive rate: {row['false_positive_rate']:.3f}",
                "",
            ]
        )

    if not best_severe.empty:
        row = best_severe.iloc[0]

        summary_lines.extend(
            [
                "Best BOD5 >= 75 detector:",
                f"  Detector: {row['detector']}",
                f"  Precision: {row['precision']:.3f}",
                f"  Recall: {row['recall']:.3f}",
                f"  F1: {row['f1']:.3f}",
                f"  False-positive rate: {row['false_positive_rate']:.3f}",
                "",
            ]
        )

    summary_lines.extend(
        [
            "Interpretation:",
            "  This analysis tests whether process measurements",
            "  contain useful abnormal-regime information.",
            "  Results should not be treated as production alarm",
            "  thresholds until validated on independent data.",
            "",
            "Deployment status:",
            "  NO-GO pending independent validation.",
            "",
        ]
    )

    SUMMARY_OUTPUT.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print()
    print("=" * 70)
    print("V2.5 PROCESS ANOMALY ANALYSIS FILES SAVED")
    print("=" * 70)

    print(COMPARISON_OUTPUT)
    print(PREDICTIONS_OUTPUT)
    print(SUMMARY_OUTPUT)

    print()
    print("=" * 70)
    print("V2.5 PROCESS ANOMALY DETECTION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()