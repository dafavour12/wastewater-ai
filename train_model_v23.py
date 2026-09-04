from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


BASE_DIR = Path(__file__).resolve().parent

RAW_DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "water-treatment.data"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "DBO-S"

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


INFLUENT_FEATURES = [
    "Q-E",
    "ZN-E",
    "PH-E",
    "DBO-E",
    "DQO-E",
    "SS-E",
    "SSV-E",
    "SED-E",
    "COND-E",
]


def safe_divide(numerator, denominator):
    """Perform safe element-wise division."""
    result = numerator / denominator.replace(0, np.nan)

    return (
        result
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )


def prepare_features(df):
    """
    Convert UCI names to GEN-BUILD names and create
    physically meaningful influent-only features.
    """

    data = df.copy()

    data = data.rename(
        columns={
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
    )

    # ---------------------------------------------------------
    # Loading features
    # ---------------------------------------------------------
    data["bod5_loading"] = (
        data["flow_m3_day"]
        * data["influent_bod5"]
    )

    data["cod_loading"] = (
        data["flow_m3_day"]
        * data["influent_cod"]
    )

    data["tss_loading"] = (
        data["flow_m3_day"]
        * data["influent_tss"]
    )

    data["vss_loading"] = (
        data["flow_m3_day"]
        * data["influent_vss"]
    )

    # ---------------------------------------------------------
    # Wastewater characteristic ratios
    # ---------------------------------------------------------
    data["cod_bod5_ratio"] = safe_divide(
        data["influent_cod"],
        data["influent_bod5"],
    )

    data["tss_bod5_ratio"] = safe_divide(
        data["influent_tss"],
        data["influent_bod5"],
    )

    data["vss_tss_ratio"] = safe_divide(
        data["influent_vss"],
        data["influent_tss"],
    )

    data["sed_tss_ratio"] = safe_divide(
        data["influent_sediments"],
        data["influent_tss"],
    )

    feature_columns = [
        "flow_m3_day",
        "influent_zinc",
        "influent_ph",
        "influent_bod5",
        "influent_cod",
        "influent_tss",
        "influent_vss",
        "influent_sediments",
        "influent_conductivity",
        "bod5_loading",
        "cod_loading",
        "tss_loading",
        "vss_loading",
        "cod_bod5_ratio",
        "tss_bod5_ratio",
        "vss_tss_ratio",
        "sed_tss_ratio",
    ]

    return data, feature_columns


def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    r2 = r2_score(
        y_true,
        y_pred,
    )

    return mae, rmse, r2


def build_models():
    return {
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=300,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        ),
        "Random Forest Tuned": RandomForestRegressor(
            n_estimators=500,
            max_depth=10,
            min_samples_leaf=2,
            max_features=0.8,
            random_state=42,
            n_jobs=-1,
        ),
    }


def evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
):
    models = build_models()

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []
    fitted_models = {}

    for name, model in models.items():

        model.fit(
            X_train,
            y_train,
        )

        test_predictions = model.predict(
            X_test
        )

        test_mae, test_rmse, test_r2 = (
            calculate_metrics(
                y_test,
                test_predictions,
            )
        )

        cv_mae = []
        cv_rmse = []
        cv_r2 = []

        for train_index, validation_index in cv.split(
            X_train
        ):
            X_cv_train = X_train.iloc[
                train_index
            ]

            X_cv_validation = X_train.iloc[
                validation_index
            ]

            y_cv_train = y_train.iloc[
                train_index
            ]

            y_cv_validation = y_train.iloc[
                validation_index
            ]

            cv_model = model.__class__(
                **model.get_params()
            )

            cv_model.fit(
                X_cv_train,
                y_cv_train,
            )

            cv_predictions = cv_model.predict(
                X_cv_validation
            )

            fold_mae, fold_rmse, fold_r2 = (
                calculate_metrics(
                    y_cv_validation,
                    cv_predictions,
                )
            )

            cv_mae.append(fold_mae)
            cv_rmse.append(fold_rmse)
            cv_r2.append(fold_r2)

        results.append(
            {
                "model": name,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "cv_mae_mean": np.mean(cv_mae),
                "cv_mae_std": np.std(
                    cv_mae,
                    ddof=1,
                ),
                "cv_rmse_mean": np.mean(cv_rmse),
                "cv_rmse_std": np.std(
                    cv_rmse,
                    ddof=1,
                ),
                "cv_r2_mean": np.mean(cv_r2),
                "cv_r2_std": np.std(
                    cv_r2,
                    ddof=1,
                ),
            }
        )

        fitted_models[name] = model

    results_df = pd.DataFrame(
        results
    ).sort_values(
        by="cv_mae_mean",
        ascending=True,
    ).reset_index(drop=True)

    best_name = results_df.iloc[0]["model"]

    return (
        results_df,
        fitted_models[best_name],
        best_name,
    )


def main():
    print("=" * 70)
    print("WASTEWATER AI - V2.3 INFLUENT FEATURE EXPANSION")
    print("=" * 70)

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_DATA_PATH}"
        )

    # ---------------------------------------------------------
    # Load complete raw UCI dataset
    # ---------------------------------------------------------
    df = pd.read_csv(
        RAW_DATA_PATH,
        header=None,
        names=UCI_COLUMNS,
        na_values=[
            "?",
            "",
            "NA",
            "N/A",
        ],
    )

    for column in UCI_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    print(
        f"\nRaw UCI observations: {len(df)}"
    )

    # ---------------------------------------------------------
    # Keep ONLY influent variables + target
    # ---------------------------------------------------------
    selected_columns = (
        INFLUENT_FEATURES
        + [TARGET]
    )

    selected_df = df[
        selected_columns
    ].copy()

    rows_before = len(selected_df)

    selected_df = selected_df.dropna(
        subset=selected_columns
    ).reset_index(
        drop=True
    )

    print(
        f"Complete influent/target rows: "
        f"{len(selected_df)}"
    )

    print(
        f"Rows removed for missing data: "
        f"{rows_before - len(selected_df)}"
    )

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------
    prepared_df, feature_columns = (
        prepare_features(
            selected_df
        )
    )

    prepared_df = prepared_df.dropna(
        subset=feature_columns
    ).reset_index(
        drop=True
    )

    X = prepared_df[
        feature_columns
    ]

    y = prepared_df[
        "effluent_bod5"
    ]

    print(
        f"Final modeling rows: {len(prepared_df)}"
    )

    print(
        f"Total modeling features: "
        f"{len(feature_columns)}"
    )

    print("\nFeatures:")
    for feature in feature_columns:
        print(f"- {feature}")

    # ---------------------------------------------------------
    # Random holdout
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.20,
            random_state=42,
        )
    )

    print("\n" + "-" * 70)
    print("RANDOM HOLDOUT VALIDATION")
    print("-" * 70)

    random_results, random_best_model, random_best_name = (
        evaluate_models(
            X_train,
            X_test,
            y_train,
            y_test,
        )
    )

    print(
        random_results.to_string(
            index=False
        )
    )

    random_predictions = (
        random_best_model.predict(
            X_test
        )
    )

    random_mae, random_rmse, random_r2 = (
        calculate_metrics(
            y_test,
            random_predictions,
        )
    )

    print(
        f"\nBest random model: "
        f"{random_best_name}"
    )

    print(
        f"Holdout MAE:  "
        f"{random_mae:.2f} mg/L"
    )

    print(
        f"Holdout RMSE: "
        f"{random_rmse:.2f} mg/L"
    )

    print(
        f"Holdout R²:   "
        f"{random_r2:.3f}"
    )

    # ---------------------------------------------------------
    # Chronological holdout
    # ---------------------------------------------------------
    split_index = int(
        len(prepared_df) * 0.80
    )

    chronological_train = (
        prepared_df.iloc[
            :split_index
        ]
    )

    chronological_test = (
        prepared_df.iloc[
            split_index:
        ]
    )

    X_chrono_train = (
        chronological_train[
            feature_columns
        ]
    )

    y_chrono_train = (
        chronological_train[
            "effluent_bod5"
        ]
    )

    X_chrono_test = (
        chronological_test[
            feature_columns
        ]
    )

    y_chrono_test = (
        chronological_test[
            "effluent_bod5"
        ]
    )

    print("\n" + "-" * 70)
    print("CHRONOLOGICAL HOLDOUT VALIDATION")
    print("-" * 70)

    print(
        f"Training rows: "
        f"{len(X_chrono_train)}"
    )

    print(
        f"Future test rows: "
        f"{len(X_chrono_test)}"
    )

    chrono_results, chrono_best_model, chrono_best_name = (
        evaluate_models(
            X_chrono_train,
            X_chrono_test,
            y_chrono_train,
            y_chrono_test,
        )
    )

    print(
        chrono_results.to_string(
            index=False
        )
    )

    chrono_predictions = (
        chrono_best_model.predict(
            X_chrono_test
        )
    )

    chrono_mae, chrono_rmse, chrono_r2 = (
        calculate_metrics(
            y_chrono_test,
            chrono_predictions,
        )
    )

    print(
        f"\nBest chronological model: "
        f"{chrono_best_name}"
    )

    print(
        f"Future-test MAE:  "
        f"{chrono_mae:.2f} mg/L"
    )

    print(
        f"Future-test RMSE: "
        f"{chrono_rmse:.2f} mg/L"
    )

    print(
        f"Future-test R²:   "
        f"{chrono_r2:.3f}"
    )

    # ---------------------------------------------------------
    # Baseline
    # ---------------------------------------------------------
    baseline_prediction = (
        y_test.mean()
    )

    baseline_predictions = [
        baseline_prediction
    ] * len(y_test)

    baseline_mae, baseline_rmse, baseline_r2 = (
        calculate_metrics(
            y_test,
            baseline_predictions,
        )
    )

    print("\n" + "-" * 70)
    print("RANDOM HOLDOUT BASELINE")
    print("-" * 70)

    print(
        f"Mean prediction: "
        f"{baseline_prediction:.2f} mg/L"
    )

    print(
        f"Baseline MAE:  "
        f"{baseline_mae:.2f} mg/L"
    )

    print(
        f"Baseline RMSE: "
        f"{baseline_rmse:.2f} mg/L"
    )

    print(
        f"Baseline R²:   "
        f"{baseline_r2:.3f}"
    )

    print(
        f"\nV2.3 MAE improvement: "
        f"{baseline_mae - random_mae:.2f} mg/L"
    )

    # ---------------------------------------------------------
    # Save model comparison
    # ---------------------------------------------------------
    random_results = random_results.copy()
    random_results["validation"] = "random"

    chrono_results = chrono_results.copy()
    chrono_results["validation"] = (
        "chronological"
    )

    comparison = pd.concat(
        [
            random_results,
            chrono_results,
        ],
        ignore_index=True,
    )

    comparison_path = (
        MODEL_DIR
        / "v23_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save chronological predictions
    # ---------------------------------------------------------
    chrono_prediction_df = (
        chronological_test[
            feature_columns
        ].copy()
    )

    chrono_prediction_df[
        "actual_effluent_bod5"
    ] = y_chrono_test.values

    chrono_prediction_df[
        "predicted_effluent_bod5"
    ] = chrono_predictions

    chrono_prediction_df[
        "absolute_error"
    ] = (
        chrono_prediction_df[
            "actual_effluent_bod5"
        ]
        - chrono_prediction_df[
            "predicted_effluent_bod5"
        ]
    ).abs()

    chrono_prediction_path = (
        MODEL_DIR
        / "v23_chronological_predictions.csv"
    )

    chrono_prediction_df.to_csv(
        chrono_prediction_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Fit final candidate on all available data
    # ---------------------------------------------------------
    final_model = (
        random_best_model.__class__(
            **random_best_model.get_params()
        )
    )

    final_model.fit(
        X,
        y,
    )

    final_model_path = (
        MODEL_DIR
        / "wastewater_bod5_model_v23.joblib"
    )

    joblib.dump(
        final_model,
        final_model_path,
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------
    if hasattr(
        final_model,
        "feature_importances_",
    ):
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance": (
                    final_model
                    .feature_importances_
                ),
            }
        ).sort_values(
            by="importance",
            ascending=False,
        )

        importance_path = (
            MODEL_DIR
            / "v23_feature_importance.csv"
        )

        importance_df.to_csv(
            importance_path,
            index=False,
        )

        print("\nFeature importance:")
        print(
            importance_df.to_string(
                index=False
            )
        )

    # ---------------------------------------------------------
    # Save feature schema
    # ---------------------------------------------------------
    schema_path = (
        MODEL_DIR
        / "v23_feature_schema.txt"
    )

    schema_path.write_text(
        "\n".join(feature_columns),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("V2.3 FILES SAVED")
    print("=" * 70)

    print(comparison_path)
    print(chrono_prediction_path)
    print(final_model_path)
    print(schema_path)

    if hasattr(
        final_model,
        "feature_importances_",
    ):
        print(importance_path)

    print(
        "\nV2.3 training completed successfully."
    )


if __name__ == "__main__":
    main()