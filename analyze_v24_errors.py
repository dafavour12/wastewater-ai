from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"

CHRONO_FILE = MODELS_DIR / "v24_chronological_predictions.csv"
RANDOM_FILE = MODELS_DIR / "v24_random_predictions.csv"

ERROR_ANALYSIS_FILE = MODELS_DIR / "v24_error_analysis.csv"
RANGE_ANALYSIS_FILE = MODELS_DIR / "v24_bod5_range_analysis.csv"
SUMMARY_FILE = MODELS_DIR / "v24_residual_summary.txt"


# ============================================================================
# HELPERS
# ============================================================================

def calculate_metrics(df):
    actual = df["actual_effluent_bod5"]
    predicted = df["predicted_effluent_bod5"]

    error = actual - predicted
    absolute_error = np.abs(error)

    mae = absolute_error.mean()
    rmse = np.sqrt((error ** 2).mean())

    ss_res = ((actual - predicted) ** 2).sum()
    ss_tot = ((actual - actual.mean()) ** 2).sum()

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

    return {
        "observations": len(df),
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mean_error": error.mean(),
        "median_error": error.median(),
        "std_error": error.std(),
        "min_error": error.min(),
        "max_error": error.max(),
        "max_absolute_error": absolute_error.max(),
    }


def analyze_ranges(df):
    """
    Analyze model performance by actual effluent BOD5 range.
    """

    conditions = [
        df["actual_effluent_bod5"] <= 20,
        df["actual_effluent_bod5"].between(21, 30),
        df["actual_effluent_bod5"].between(31, 50),
        df["actual_effluent_bod5"] > 50,
    ]

    labels = [
        "low_<=20",
        "moderate_21_30",
        "high_31_50",
        "very_high_>50",
    ]

    result_rows = []

    for condition, label in zip(conditions, labels):

        subset = df[condition].copy()

        if len(subset) == 0:
            result_rows.append(
                {
                    "bod5_range": label,
                    "observations": 0,
                    "mae": np.nan,
                    "rmse": np.nan,
                    "mean_error": np.nan,
                    "median_error": np.nan,
                    "minimum_actual": np.nan,
                    "maximum_actual": np.nan,
                }
            )
            continue

        actual = subset["actual_effluent_bod5"]
        predicted = subset["predicted_effluent_bod5"]

        error = actual - predicted

        result_rows.append(
            {
                "bod5_range": label,
                "observations": len(subset),
                "mae": np.abs(error).mean(),
                "rmse": np.sqrt((error ** 2).mean()),
                "mean_error": error.mean(),
                "median_error": error.median(),
                "minimum_actual": actual.min(),
                "maximum_actual": actual.max(),
            }
        )

    return pd.DataFrame(result_rows)


def analyze_dataset(df, dataset_name):
    """
    Create observation-level error analysis.
    """

    result = df.copy()

    result["dataset"] = dataset_name

    result["error"] = (
        result["actual_effluent_bod5"]
        - result["predicted_effluent_bod5"]
    )

    result["absolute_error"] = np.abs(
        result["error"]
    )

    result["percentage_error"] = np.where(
        result["actual_effluent_bod5"] != 0,
        (
            result["error"]
            / result["actual_effluent_bod5"]
        )
        * 100,
        np.nan,
    )

    result["absolute_percentage_error"] = np.abs(
        result["percentage_error"]
    )

    result["prediction_direction"] = np.where(
        result["error"] > 0,
        "underprediction",
        np.where(
            result["error"] < 0,
            "overprediction",
            "exact",
        ),
    )

    return result


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print(
        "WASTEWATER AI - V2.4 ROBUSTNESS & ERROR ANALYSIS"
    )
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Validate files
    # ------------------------------------------------------------------------

    if not CHRONO_FILE.exists():
        raise FileNotFoundError(
            f"Missing file:\n{CHRONO_FILE}"
        )

    if not RANDOM_FILE.exists():
        raise FileNotFoundError(
            f"Missing file:\n{RANDOM_FILE}"
        )

    # ------------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------------

    chrono = pd.read_csv(
        CHRONO_FILE
    )

    random = pd.read_csv(
        RANDOM_FILE
    )

    chrono = analyze_dataset(
        chrono,
        "chronological",
    )

    random = analyze_dataset(
        random,
        "random",
    )

    # =========================================================================
    # CHRONOLOGICAL ANALYSIS
    # =========================================================================

    print("\n" + "-" * 70)
    print(
        "CHRONOLOGICAL HOLDOUT"
    )
    print("-" * 70)

    chrono_metrics = calculate_metrics(
        chrono
    )

    print(
        f"Observations: "
        f"{chrono_metrics['observations']}"
    )

    print(
        f"MAE: "
        f"{chrono_metrics['mae']:.2f} mg/L"
    )

    print(
        f"RMSE: "
        f"{chrono_metrics['rmse']:.2f} mg/L"
    )

    print(
        f"R²: "
        f"{chrono_metrics['r2']:.3f}"
    )

    print(
        f"Mean signed error: "
        f"{chrono_metrics['mean_error']:.2f} mg/L"
    )

    print(
        f"Median signed error: "
        f"{chrono_metrics['median_error']:.2f} mg/L"
    )

    print(
        f"Maximum absolute error: "
        f"{chrono_metrics['max_absolute_error']:.2f} mg/L"
    )

    # ------------------------------------------------------------------------
    # Directional bias
    # ------------------------------------------------------------------------

    underpredictions = (
        chrono["prediction_direction"]
        == "underprediction"
    ).sum()

    overpredictions = (
        chrono["prediction_direction"]
        == "overprediction"
    ).sum()

    exact_predictions = (
        chrono["prediction_direction"]
        == "exact"
    ).sum()

    print("\nPrediction direction:")

    print(
        f"Underpredictions: {underpredictions}"
    )

    print(
        f"Overpredictions: {overpredictions}"
    )

    print(
        f"Exact: {exact_predictions}"
    )

    # =========================================================================
    # WORST CHRONOLOGICAL PREDICTIONS
    # =========================================================================

    print("\nWorst chronological predictions:")

    worst_chrono = (
        chrono
        .sort_values(
            "absolute_error",
            ascending=False,
        )
        .head(10)
    )

    print(
        worst_chrono[
            [
                "actual_effluent_bod5",
                "predicted_effluent_bod5",
                "error",
                "absolute_error",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # BOD5 RANGE ANALYSIS
    # =========================================================================

    print("\nBOD5 performance by range:")

    chrono_ranges = analyze_ranges(
        chrono
    )

    print(
        chrono_ranges.to_string(
            index=False
        )
    )

    # =========================================================================
    # RANDOM ANALYSIS
    # =========================================================================

    print("\n" + "-" * 70)
    print(
        "RANDOM HOLDOUT"
    )
    print("-" * 70)

    random_metrics = calculate_metrics(
        random
    )

    print(
        f"Observations: "
        f"{random_metrics['observations']}"
    )

    print(
        f"MAE: "
        f"{random_metrics['mae']:.2f} mg/L"
    )

    print(
        f"RMSE: "
        f"{random_metrics['rmse']:.2f} mg/L"
    )

    print(
        f"R²: "
        f"{random_metrics['r2']:.3f}"
    )

    print(
        f"Mean signed error: "
        f"{random_metrics['mean_error']:.2f} mg/L"
    )

    print(
        f"Median signed error: "
        f"{random_metrics['median_error']:.2f} mg/L"
    )

    print(
        f"Maximum absolute error: "
        f"{random_metrics['max_absolute_error']:.2f} mg/L"
    )

    # =========================================================================
    # WORST RANDOM PREDICTIONS
    # =========================================================================

    print("\nWorst random predictions:")

    worst_random = (
        random
        .sort_values(
            "absolute_error",
            ascending=False,
        )
        .head(10)
    )

    print(
        worst_random[
            [
                "actual_effluent_bod5",
                "predicted_effluent_bod5",
                "error",
                "absolute_error",
            ]
        ].to_string(index=False)
    )

    # =========================================================================
    # COMBINED ERROR ANALYSIS
    # =========================================================================

    combined = pd.concat(
        [
            chrono,
            random,
        ],
        ignore_index=True,
    )

    combined[
        [
            "dataset",
            "actual_effluent_bod5",
            "predicted_effluent_bod5",
            "error",
            "absolute_error",
            "percentage_error",
            "absolute_percentage_error",
            "prediction_direction",
        ]
    ].to_csv(
        ERROR_ANALYSIS_FILE,
        index=False,
    )

    # =========================================================================
    # RANGE ANALYSIS
    # =========================================================================

    chrono_ranges["dataset"] = "chronological"
    random_ranges = analyze_ranges(
        random
    )
    random_ranges["dataset"] = "random"

    range_analysis = pd.concat(
        [
            chrono_ranges,
            random_ranges,
        ],
        ignore_index=True,
    )

    range_analysis.to_csv(
        RANGE_ANALYSIS_FILE,
        index=False,
    )

    # =========================================================================
    # RESIDUAL SUMMARY
    # =========================================================================

    with open(
        SUMMARY_FILE,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "WASTEWATER AI V2.4 RESIDUAL SUMMARY\n"
        )

        file.write(
            "=" * 55 + "\n\n"
        )

        file.write(
            "CHRONOLOGICAL HOLDOUT\n"
        )

        file.write(
            f"Observations: "
            f"{chrono_metrics['observations']}\n"
        )

        file.write(
            f"MAE: "
            f"{chrono_metrics['mae']:.4f} mg/L\n"
        )

        file.write(
            f"RMSE: "
            f"{chrono_metrics['rmse']:.4f} mg/L\n"
        )

        file.write(
            f"R2: "
            f"{chrono_metrics['r2']:.4f}\n"
        )

        file.write(
            f"Mean signed error: "
            f"{chrono_metrics['mean_error']:.4f} mg/L\n"
        )

        file.write(
            f"Median signed error: "
            f"{chrono_metrics['median_error']:.4f} mg/L\n"
        )

        file.write(
            f"Maximum absolute error: "
            f"{chrono_metrics['max_absolute_error']:.4f} mg/L\n\n"
        )

        file.write(
            "Prediction direction:\n"
        )

        file.write(
            f"Underpredictions: {underpredictions}\n"
        )

        file.write(
            f"Overpredictions: {overpredictions}\n"
        )

        file.write(
            f"Exact: {exact_predictions}\n\n"
        )

        file.write(
            "RANDOM HOLDOUT\n"
        )

        file.write(
            f"Observations: "
            f"{random_metrics['observations']}\n"
        )

        file.write(
            f"MAE: "
            f"{random_metrics['mae']:.4f} mg/L\n"
        )

        file.write(
            f"RMSE: "
            f"{random_metrics['rmse']:.4f} mg/L\n"
        )

        file.write(
            f"R2: "
            f"{random_metrics['r2']:.4f}\n"
        )

        file.write(
            f"Mean signed error: "
            f"{random_metrics['mean_error']:.4f} mg/L\n"
        )

        file.write(
            f"Median signed error: "
            f"{random_metrics['median_error']:.4f} mg/L\n"
        )

        file.write(
            f"Maximum absolute error: "
            f"{random_metrics['max_absolute_error']:.4f} mg/L\n\n"
        )

        file.write(
            "INTERPRETATION NOTES\n"
        )

        if chrono_metrics["mean_error"] < 0:
            file.write(
                "The chronological model tends to overpredict "
                "effluent BOD5.\n"
            )
        elif chrono_metrics["mean_error"] > 0:
            file.write(
                "The chronological model tends to underpredict "
                "effluent BOD5.\n"
            )
        else:
            file.write(
                "The chronological model has approximately zero "
                "mean signed error.\n"
            )

        if chrono_metrics["r2"] > 0:
            file.write(
                "Chronological R2 is positive, indicating improvement "
                "over the mean predictor on this holdout.\n"
            )
        else:
            file.write(
                "Chronological R2 is negative, indicating performance "
                "below the mean predictor.\n"
            )

        file.write(
            "Final deployment decisions must consider error by "
            "operating range, not MAE alone.\n"
        )

    # =========================================================================
    # FINAL SUMMARY
    # =========================================================================

    print("\n" + "=" * 70)
    print(
        "V2.4 ERROR ANALYSIS FILES SAVED"
    )
    print("=" * 70)

    print(
        ERROR_ANALYSIS_FILE
    )

    print(
        RANGE_ANALYSIS_FILE
    )

    print(
        SUMMARY_FILE
    )

    print("\n" + "=" * 70)
    print(
        "V2.4 ROBUSTNESS ANALYSIS COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
