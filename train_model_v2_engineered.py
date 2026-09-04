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

DATA_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "wastewater_v2.csv"
)

MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

TARGET = "effluent_bod5"

BASE_FEATURES = [
    "flow_m3_day",
    "influent_ph",
    "influent_bod5",
    "influent_cod",
    "influent_tss",
]


def safe_divide(numerator, denominator):
    """Divide safely and replace invalid values with zero."""
    result = numerator / denominator.replace(0, np.nan)
    return result.replace([np.inf, -np.inf], np.nan).fillna(0)


def engineer_features(df):
    """
    Create physically meaningful wastewater process features.

    All engineered variables use influent/process information only.
    No effluent variables are used.
    """
    data = df.copy()

    # Loading-related features.
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

    # Ratios describing wastewater characteristics.
    data["cod_bod5_ratio"] = safe_divide(
        data["influent_cod"],
        data["influent_bod5"],
    )

    data["tss_bod5_ratio"] = safe_divide(
        data["influent_tss"],
        data["influent_bod5"],
    )

    data["cod_tss_ratio"] = safe_divide(
        data["influent_cod"],
        data["influent_tss"],
    )

    return data


ENGINEERED_FEATURES = [
    "flow_m3_day",
    "influent_ph",
    "influent_bod5",
    "influent_cod",
    "influent_tss",
    "bod5_loading",
    "cod_loading",
    "tss_loading",
    "cod_bod5_ratio",
    "tss_bod5_ratio",
    "cod_tss_ratio",
]


def metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def evaluate_models(X_train, X_test, y_train, y_test):
    models = {
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

    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        test_mae, test_rmse, test_r2 = metrics(
            y_test,
            predictions,
        )

        cv_mae = []
        cv_rmse = []
        cv_r2 = []

        for train_idx, val_idx in cv.split(X_train):
            X_cv_train = X_train.iloc[train_idx]
            X_cv_val = X_train.iloc[val_idx]

            y_cv_train = y_train.iloc[train_idx]
            y_cv_val = y_train.iloc[val_idx]

            cv_model = model.__class__(
                **model.get_params()
            )

            cv_model.fit(
                X_cv_train,
                y_cv_train,
            )

            cv_predictions = cv_model.predict(
                X_cv_val
            )

            fold_mae, fold_rmse, fold_r2 = metrics(
                y_cv_val,
                cv_predictions,
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
                "cv_mae_std": np.std(cv_mae, ddof=1),
                "cv_rmse_mean": np.mean(cv_rmse),
                "cv_rmse_std": np.std(cv_rmse, ddof=1),
                "cv_r2_mean": np.mean(cv_r2),
                "cv_r2_std": np.std(cv_r2, ddof=1),
            }
        )

        fitted_models[name] = model

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
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
    print("WASTEWATER AI - V2 FEATURE ENGINEERING")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = BASE_FEATURES + [TARGET]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    df = df[required_columns].dropna().copy()

    # ---------------------------------------------------------
    # Feature engineering
    # ---------------------------------------------------------
    df = engineer_features(df)

    df = df.dropna().reset_index(drop=True)

    X = df[ENGINEERED_FEATURES]
    y = df[TARGET]

    print(f"\nDataset rows: {len(df)}")

    print("\nEngineered features:")
    for feature in ENGINEERED_FEATURES:
        print(f"- {feature}")

    # ---------------------------------------------------------
    # Random holdout
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
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
        random_results.to_string(index=False)
    )

    random_best_pred = random_best_model.predict(
        X_test
    )

    random_mae, random_rmse, random_r2 = metrics(
        y_test,
        random_best_pred,
    )

    print(f"\nBest random model: {random_best_name}")
    print(f"Holdout MAE:  {random_mae:.2f} mg/L")
    print(f"Holdout RMSE: {random_rmse:.2f} mg/L")
    print(f"Holdout R²:   {random_r2:.3f}")

    # ---------------------------------------------------------
    # Chronological holdout
    # ---------------------------------------------------------
    split_index = int(len(df) * 0.80)

    chronological_train = df.iloc[:split_index]
    chronological_test = df.iloc[split_index:]

    X_chrono_train = chronological_train[
        ENGINEERED_FEATURES
    ]

    y_chrono_train = chronological_train[
        TARGET
    ]

    X_chrono_test = chronological_test[
        ENGINEERED_FEATURES
    ]

    y_chrono_test = chronological_test[
        TARGET
    ]

    print("\n" + "-" * 70)
    print("CHRONOLOGICAL HOLDOUT VALIDATION")
    print("-" * 70)

    print(
        f"Training rows: {len(X_chrono_train)}"
    )

    print(
        f"Future test rows: {len(X_chrono_test)}"
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
        chrono_results.to_string(index=False)
    )

    chrono_best_pred = chrono_best_model.predict(
        X_chrono_test
    )

    chrono_mae, chrono_rmse, chrono_r2 = metrics(
        y_chrono_test,
        chrono_best_pred,
    )

    print(f"\nBest chronological model: {chrono_best_name}")
    print(f"Future-test MAE:  {chrono_mae:.2f} mg/L")
    print(f"Future-test RMSE: {chrono_rmse:.2f} mg/L")
    print(f"Future-test R²:   {chrono_r2:.3f}")

    # ---------------------------------------------------------
    # Save validation results
    # ---------------------------------------------------------
    random_results = random_results.copy()
    random_results["validation"] = "random"

    chrono_results = chrono_results.copy()
    chrono_results["validation"] = "chronological"

    comparison = pd.concat(
        [
            random_results,
            chrono_results,
        ],
        ignore_index=True,
    )

    comparison_path = (
        MODEL_DIR
        / "v2_engineered_model_comparison.csv"
    )

    comparison.to_csv(
        comparison_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Save chronological predictions
    # ---------------------------------------------------------
    chronological_predictions = (
        chronological_test.copy()
    )

    chronological_predictions[
        "actual_effluent_bod5"
    ] = y_chrono_test.values

    chronological_predictions[
        "predicted_effluent_bod5"
    ] = chrono_best_pred

    chronological_predictions[
        "absolute_error"
    ] = (
        chronological_predictions[
            "actual_effluent_bod5"
        ]
        - chronological_predictions[
            "predicted_effluent_bod5"
        ]
    ).abs()

    chrono_prediction_path = (
        MODEL_DIR
        / "v2_engineered_chronological_predictions.csv"
    )

    chronological_predictions.to_csv(
        chrono_prediction_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Fit final production candidate on all data
    # ---------------------------------------------------------
    final_model = random_best_model.__class__(
        **random_best_model.get_params()
    )

    final_model.fit(X, y)

    final_model_path = (
        MODEL_DIR
        / "wastewater_bod5_model_v2_engineered.joblib"
    )

    joblib.dump(
        final_model,
        final_model_path,
    )

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------
    if hasattr(final_model, "feature_importances_"):
        importance = pd.DataFrame(
            {
                "feature": ENGINEERED_FEATURES,
                "importance": (
                    final_model.feature_importances_
                ),
            }
        ).sort_values(
            by="importance",
            ascending=False,
        )

        importance_path = (
            MODEL_DIR
            / "v2_engineered_feature_importance.csv"
        )

        importance.to_csv(
            importance_path,
            index=False,
        )

        print("\nFeature importance:")
        print(
            importance.to_string(index=False)
        )

    print("\n" + "=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(comparison_path)
    print(chrono_prediction_path)
    print(final_model_path)

    if hasattr(final_model, "feature_importances_"):
        print(importance_path)

    print("\nFeature-engineered V2 training completed.")


if __name__ == "__main__":
    main()