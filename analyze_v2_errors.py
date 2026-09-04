from pathlib import Path

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


BASE_DIR = Path(__file__).resolve().parent

PREDICTIONS_PATH = (
    BASE_DIR
    / "models"
    / "v2_test_predictions.csv"
)

OUTPUT_PATH = (
    BASE_DIR
    / "models"
    / "v2_error_analysis.csv"
)


def main():
    print("=" * 70)
    print("WASTEWATER AI - V2 HOLDOUT ERROR ANALYSIS")
    print("=" * 70)

    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(
            f"Holdout predictions not found: {PREDICTIONS_PATH}"
        )

    df = pd.read_csv(PREDICTIONS_PATH)

    required_columns = [
        "actual_effluent_bod5",
        "predicted_effluent_bod5",
        "absolute_error",
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    actual = df["actual_effluent_bod5"]
    predicted = df["predicted_effluent_bod5"]

    # ---------------------------------------------------------
    # Overall holdout performance
    # ---------------------------------------------------------
    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    r2 = r2_score(actual, predicted)

    print("\nHoldout performance")
    print("-" * 70)
    print(f"Observations: {len(df)}")
    print(f"MAE:          {mae:.2f} mg/L")
    print(f"RMSE:         {rmse:.2f} mg/L")
    print(f"R²:           {r2:.3f}")

    # ---------------------------------------------------------
    # Baseline comparison
    # ---------------------------------------------------------
    baseline_prediction = actual.mean()

    baseline_mae = mean_absolute_error(
        actual,
        [baseline_prediction] * len(actual),
    )

    baseline_rmse = mean_squared_error(
        actual,
        [baseline_prediction] * len(actual),
    ) ** 0.5

    baseline_r2 = r2_score(
        actual,
        [baseline_prediction] * len(actual),
    )

    print("\nMean baseline")
    print("-" * 70)
    print(f"Baseline prediction: {baseline_prediction:.2f} mg/L")
    print(f"Baseline MAE:        {baseline_mae:.2f} mg/L")
    print(f"Baseline RMSE:       {baseline_rmse:.2f} mg/L")
    print(f"Baseline R²:         {baseline_r2:.3f}")

    print("\nModel vs baseline")
    print("-" * 70)

    mae_difference = baseline_mae - mae
    rmse_difference = baseline_rmse - rmse

    print(
        f"MAE improvement:  {mae_difference:.2f} mg/L"
    )
    print(
        f"RMSE improvement: {rmse_difference:.2f} mg/L"
    )

    if mae_difference > 0:
        print("Result: Model improves on baseline MAE.")
    else:
        print("Result: Model does NOT improve on baseline MAE.")

    # ---------------------------------------------------------
    # Error statistics
    # ---------------------------------------------------------
    df["error"] = (
        df["actual_effluent_bod5"]
        - df["predicted_effluent_bod5"]
    )

    df["absolute_error"] = df["error"].abs()

    df["percentage_error"] = (
        df["absolute_error"]
        / df["actual_effluent_bod5"].replace(0, pd.NA)
        * 100
    )

    print("\nError statistics")
    print("-" * 70)
    print(df["error"].describe().to_string())

    # ---------------------------------------------------------
    # Worst predictions
    # ---------------------------------------------------------
    print("\nWorst 10 predictions")
    print("-" * 70)

    worst = df.sort_values(
        by="absolute_error",
        ascending=False,
    ).head(10)

    display_columns = [
        "actual_effluent_bod5",
        "predicted_effluent_bod5",
        "error",
        "absolute_error",
        "percentage_error",
    ]

    print(
        worst[display_columns].to_string(index=False)
    )

    # ---------------------------------------------------------
    # Target distribution
    # ---------------------------------------------------------
    print("\nHoldout target distribution")
    print("-" * 70)
    print(actual.describe().to_string())

    # ---------------------------------------------------------
    # Performance by BOD5 range
    # ---------------------------------------------------------
    bins = [
        -float("inf"),
        20,
        30,
        50,
        100,
        float("inf"),
    ]

    labels = [
        "low_<=20",
        "moderate_21_30",
        "high_31_50",
        "very_high_51_100",
        "extreme_>100",
    ]

    df["bod5_range"] = pd.cut(
        df["actual_effluent_bod5"],
        bins=bins,
        labels=labels,
    )

    range_results = []

    for group, group_df in df.groupby(
        "bod5_range",
        observed=False,
    ):
        if len(group_df) == 0:
            continue

        range_results.append(
            {
                "bod5_range": str(group),
                "count": len(group_df),
                "mae": mean_absolute_error(
                    group_df["actual_effluent_bod5"],
                    group_df["predicted_effluent_bod5"],
                ),
                "rmse": mean_squared_error(
                    group_df["actual_effluent_bod5"],
                    group_df["predicted_effluent_bod5"],
                ) ** 0.5,
            }
        )

    range_df = pd.DataFrame(range_results)

    print("\nPerformance by BOD5 range")
    print("-" * 70)
    print(range_df.to_string(index=False))

    # ---------------------------------------------------------
    # Save detailed analysis
    # ---------------------------------------------------------
    df = df.sort_values(
        by="absolute_error",
        ascending=False,
    )

    df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print("\nSaved:")
    print(OUTPUT_PATH)

    print("\nError analysis completed successfully.")


if __name__ == "__main__":
    main()
