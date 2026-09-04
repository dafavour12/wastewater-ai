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
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
RAW_DATA = BASE_DIR / "data" / "raw" / "water-treatment.data"
MODELS_DIR = BASE_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
N_SPLITS = 5
TEST_SIZE = 0.20


# ============================================================================
# UCI DATASET COLUMNS
# ============================================================================

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


# ============================================================================
# FEATURE DEFINITIONS
# ============================================================================

RAW_FEATURE_MAP = {
    "Q-E": "flow_m3_day",
    "ZN-E": "influent_zinc",
    "PH-E": "influent_ph",
    "DBO-E": "influent_bod5",
    "DQO-E": "influent_cod",
    "SS-E": "influent_tss",
    "SSV-E": "influent_vss",
    "SED-E": "influent_sediments",
    "COND-E": "influent_conductivity",
}

TARGET_COLUMN = "DBO-S"

RAW_FEATURES = [
    "flow_m3_day",
    "influent_zinc",
    "influent_ph",
    "influent_bod5",
    "influent_cod",
    "influent_tss",
    "influent_vss",
    "influent_sediments",
    "influent_conductivity",
]

LOADING_FEATURES = [
    "bod5_loading",
    "cod_loading",
    "tss_loading",
    "vss_loading",
]

RATIO_FEATURES = [
    "cod_bod5_ratio",
    "tss_bod5_ratio",
    "vss_tss_ratio",
    "sed_tss_ratio",
]

ENGINEERED_FEATURES = LOADING_FEATURES + RATIO_FEATURES

ALL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES


# ============================================================================
# SAFE FEATURE ENGINEERING
# ============================================================================

def safe_divide(numerator, denominator):
    """
    Divide two series while preventing division-by-zero and invalid values.
    """
    denominator = denominator.replace(0, np.nan)

    result = numerator / denominator

    result = result.replace([np.inf, -np.inf], np.nan)

    return result


def create_features(df):
    """
    Create engineered features using only influent measurements.

    No downstream/process/effluent variables are used as predictors.
    """

    result = df.copy()

    # Loading estimates
    result["bod5_loading"] = (
        result["influent_bod5"] * result["flow_m3_day"]
    )

    result["cod_loading"] = (
        result["influent_cod"] * result["flow_m3_day"]
    )

    result["tss_loading"] = (
        result["influent_tss"] * result["flow_m3_day"]
    )

    result["vss_loading"] = (
        result["influent_vss"] * result["flow_m3_day"]
    )

    # Concentration / composition ratios
    result["cod_bod5_ratio"] = safe_divide(
        result["influent_cod"],
        result["influent_bod5"],
    )

    result["tss_bod5_ratio"] = safe_divide(
        result["influent_tss"],
        result["influent_bod5"],
    )

    result["vss_tss_ratio"] = safe_divide(
        result["influent_vss"],
        result["influent_tss"],
    )

    result["sed_tss_ratio"] = safe_divide(
        result["influent_sediments"],
        result["influent_tss"],
    )

    return result


# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

def build_models():
    """
    Return the candidate regression models.
    """

    return {
        "Random Forest": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=400,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    ExtraTreesRegressor(
                        n_estimators=400,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                        min_samples_leaf=2,
                    ),
                ),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    GradientBoostingRegressor(
                        n_estimators=250,
                        learning_rate=0.03,
                        max_depth=2,
                        min_samples_leaf=5,
                        loss="huber",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "Random Forest Tuned": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(strategy="median"),
                ),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=600,
                        max_depth=8,
                        min_samples_split=5,
                        min_samples_leaf=3,
                        max_features=0.8,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


# ============================================================================
# METRICS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics.
    """

    mae = mean_absolute_error(y_true, y_pred)

    rmse = np.sqrt(
        mean_squared_error(y_true, y_pred)
    )

    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


# ============================================================================
# TIME-SERIES CROSS VALIDATION
# ============================================================================

def time_series_cv(model, X, y):
    """
    Perform leakage-safe chronological cross-validation.

    Each fold trains on earlier observations and validates on later
    observations.
    """

    splitter = TimeSeriesSplit(
        n_splits=N_SPLITS
    )

    fold_results = []

    for fold_number, (train_idx, validation_idx) in enumerate(
        splitter.split(X),
        start=1,
    ):
        X_train = X.iloc[train_idx]
        X_validation = X.iloc[validation_idx]

        y_train = y.iloc[train_idx]
        y_validation = y.iloc[validation_idx]

        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_validation
        )

        mae, rmse, r2 = calculate_metrics(
            y_validation,
            predictions,
        )

        fold_results.append(
            {
                "fold": fold_number,
                "mae": mae,
                "rmse": rmse,
                "r2": r2,
                "train_rows": len(train_idx),
                "validation_rows": len(validation_idx),
            }
        )

    return pd.DataFrame(fold_results)


# ============================================================================
# CHRONOLOGICAL HOLDOUT
# ============================================================================

def chronological_holdout(X, y):
    """
    Create an 80/20 chronological train/test split.

    The final 20% is treated as future/unseen data.
    """

    split_index = int(
        len(X) * (1 - TEST_SIZE)
    )

    X_train = X.iloc[:split_index].copy()
    X_test = X.iloc[split_index:].copy()

    y_train = y.iloc[:split_index].copy()
    y_test = y.iloc[split_index:].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================================
# RANDOM HOLDOUT
# ============================================================================

def random_holdout(X, y):
    """
    Create a reproducible random 80/20 split.

    This is retained as a secondary diagnostic only.
    """

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    indices = np.arange(len(X))

    rng.shuffle(indices)

    split_index = int(
        len(indices) * (1 - TEST_SIZE)
    )

    train_idx = indices[:split_index]
    test_idx = indices[split_index:]

    X_train = X.iloc[train_idx].copy()
    X_test = X.iloc[test_idx].copy()

    y_train = y.iloc[train_idx].copy()
    y_test = y.iloc[test_idx].copy()

    return (
        X_train,
        X_test,
        y_train,
        y_test,
    )


# ============================================================================
# MODEL EVALUATION
# ============================================================================

def evaluate_models(
    X_train,
    X_test,
    y_train,
    y_test,
    validation_type,
):
    """
    Evaluate all candidate models.

    Model selection is based on TimeSeriesSplit CV MAE.

    The holdout is never used for model selection.
    """

    results = []
    fitted_models = {}

    models = build_models()

    for model_name, model in models.items():

        cv_results = time_series_cv(
            model,
            X_train,
            y_train,
        )

        cv_mae_mean = cv_results["mae"].mean()
        cv_mae_std = cv_results["mae"].std(ddof=0)

        cv_rmse_mean = cv_results["rmse"].mean()
        cv_rmse_std = cv_results["rmse"].std(ddof=0)

        cv_r2_mean = cv_results["r2"].mean()
        cv_r2_std = cv_results["r2"].std(ddof=0)

        # Fit on training partition only.
        model.fit(
            X_train,
            y_train,
        )

        predictions = model.predict(
            X_test
        )

        test_mae, test_rmse, test_r2 = calculate_metrics(
            y_test,
            predictions,
        )

        results.append(
            {
                "validation_type": validation_type,
                "model": model_name,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "cv_mae_mean": cv_mae_mean,
                "cv_mae_std": cv_mae_std,
                "cv_rmse_mean": cv_rmse_mean,
                "cv_rmse_std": cv_rmse_std,
                "cv_r2_mean": cv_r2_mean,
                "cv_r2_std": cv_r2_std,
            }
        )

        fitted_models[model_name] = model

    results_df = pd.DataFrame(
        results
    )

    # Select model ONLY by CV MAE.
    best_model_name = (
        results_df
        .sort_values("cv_mae_mean")
        .iloc[0]["model"]
    )

    return (
        results_df,
        fitted_models,
        best_model_name,
    )


# ============================================================================
# BASELINE
# ============================================================================

def evaluate_mean_baseline(
    y_train,
    y_test,
):
    """
    Evaluate a simple mean predictor.
    """

    mean_prediction = float(
        y_train.mean()
    )

    predictions = np.full(
        len(y_test),
        mean_prediction,
    )

    mae, rmse, r2 = calculate_metrics(
        y_test,
        predictions,
    )

    return {
        "mean_prediction": mean_prediction,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# ============================================================================
# ABLATION STUDY
# ============================================================================

def run_ablation_study(
    df,
    y,
):
    """
    Compare feature groups using chronological TimeSeriesSplit.

    Experiments:
        A - raw influent features
        B - raw + loading features
        C - raw + ratio features
        D - all features
    """

    feature_sets = {
        "A_raw_only": RAW_FEATURES,
        "B_raw_plus_loading": RAW_FEATURES + LOADING_FEATURES,
        "C_raw_plus_ratios": RAW_FEATURES + RATIO_FEATURES,
        "D_all_features": ALL_FEATURES,
    }

    rows = []

    for experiment_name, features in feature_sets.items():

        X = df[features].copy()

        models = build_models()

        for model_name, model in models.items():

            cv_results = time_series_cv(
                model,
                X,
                y,
            )

            rows.append(
                {
                    "experiment": experiment_name,
                    "feature_count": len(features),
                    "model": model_name,
                    "cv_mae_mean": cv_results["mae"].mean(),
                    "cv_mae_std": cv_results["mae"].std(ddof=0),
                    "cv_rmse_mean": cv_results["rmse"].mean(),
                    "cv_rmse_std": cv_results["rmse"].std(ddof=0),
                    "cv_r2_mean": cv_results["r2"].mean(),
                    "cv_r2_std": cv_results["r2"].std(ddof=0),
                }
            )

    return pd.DataFrame(rows)


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def extract_feature_importance(
    fitted_model,
    feature_names,
):
    """
    Extract feature importance from tree-based model pipeline.
    """

    model = fitted_model.named_steps["model"]

    if not hasattr(
        model,
        "feature_importances_",
    ):
        return pd.DataFrame(
            columns=[
                "feature",
                "importance",
            ]
        )

    importance = model.feature_importances_

    result = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": importance,
        }
    )

    return result.sort_values(
        "importance",
        ascending=False,
    ).reset_index(
        drop=True
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print("=" * 70)
    print(
        "WASTEWATER AI - V2.4 TEMPORAL VALIDATION"
    )
    print("=" * 70)

    # ------------------------------------------------------------------------
    # Load raw UCI data
    # ------------------------------------------------------------------------

    if not RAW_DATA.exists():
        raise FileNotFoundError(
            f"Raw dataset not found:\n{RAW_DATA}"
        )

    df = pd.read_csv(
        RAW_DATA,
        header=None,
        names=UCI_COLUMNS,
        na_values=[
            "?",
            "",
            "NA",
            "N/A",
        ],
    )

    print(
        f"\nRaw UCI observations: {len(df)}"
    )

    # Convert everything to numeric.
    for column in UCI_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    # ------------------------------------------------------------------------
    # Select ONLY influent variables + target
    # ------------------------------------------------------------------------

    selected_columns = list(
        RAW_FEATURE_MAP.keys()
    ) + [TARGET_COLUMN]

    df = df[selected_columns].copy()

    df = df.rename(
        columns=RAW_FEATURE_MAP
    )

    df = df.rename(
        columns={
            TARGET_COLUMN: "effluent_bod5"
        }
    )

    before_drop = len(df)

    df = df.dropna(
        subset=RAW_FEATURES + ["effluent_bod5"]
    ).reset_index(
        drop=True
    )

    after_drop = len(df)

    print(
        f"Complete influent/target rows: {after_drop}"
    )

    print(
        f"Rows removed for missing data: "
        f"{before_drop - after_drop}"
    )

    # ------------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------------

    df = create_features(
        df
    )

    print(
        f"Final modeling rows: {len(df)}"
    )

    print(
        f"Total modeling features: {len(ALL_FEATURES)}"
    )

    print("\nRaw features:")

    for feature in RAW_FEATURES:
        print(
            f"- {feature}"
        )

    print("\nEngineered features:")

    for feature in ENGINEERED_FEATURES:
        print(
            f"- {feature}"
        )

    # ------------------------------------------------------------------------
    # Remove rows where engineered features are invalid.
    # ------------------------------------------------------------------------

    df = df.replace(
        [np.inf, -np.inf],
        np.nan,
    )

    df = df.dropna(
        subset=ALL_FEATURES + ["effluent_bod5"]
    ).reset_index(
        drop=True
    )

    X = df[ALL_FEATURES].copy()

    y = df["effluent_bod5"].copy()

    print(
        f"\nFinal valid modeling rows after feature engineering: "
        f"{len(df)}"
    )

    # =========================================================================
    # CHRONOLOGICAL HOLDOUT
    # =========================================================================

    print("\n" + "-" * 70)
    print(
        "CHRONOLOGICAL HOLDOUT VALIDATION"
    )
    print("-" * 70)

    (
        X_train_chrono,
        X_test_chrono,
        y_train_chrono,
        y_test_chrono,
    ) = chronological_holdout(
        X,
        y,
    )

    print(
        f"Training rows: {len(X_train_chrono)}"
    )

    print(
        f"Future test rows: {len(X_test_chrono)}"
    )

    (
        chrono_results,
        chrono_models,
        chrono_best,
    ) = evaluate_models(
        X_train_chrono,
        X_test_chrono,
        y_train_chrono,
        y_test_chrono,
        "chronological",
    )

    print(
        chrono_results.to_string(
            index=False
        )
    )

    print(
        f"\nBest chronological model by "
        f"TimeSeriesSplit CV MAE: {chrono_best}"
    )

    best_chrono_row = chrono_results[
        chrono_results["model"] == chrono_best
    ].iloc[0]

    print(
        f"Future-test MAE:  "
        f"{best_chrono_row['test_mae']:.2f} mg/L"
    )

    print(
        f"Future-test RMSE: "
        f"{best_chrono_row['test_rmse']:.2f} mg/L"
    )

    print(
        f"Future-test R²:   "
        f"{best_chrono_row['test_r2']:.3f}"
    )

    # ------------------------------------------------------------------------
    # Chronological baseline
    # ------------------------------------------------------------------------

    chrono_baseline = evaluate_mean_baseline(
        y_train_chrono,
        y_test_chrono,
    )

    print(
        "\nChronological mean baseline:"
    )

    print(
        f"Mean prediction: "
        f"{chrono_baseline['mean_prediction']:.2f} mg/L"
    )

    print(
        f"Baseline MAE: "
        f"{chrono_baseline['mae']:.2f} mg/L"
    )

    print(
        f"Baseline RMSE: "
        f"{chrono_baseline['rmse']:.2f} mg/L"
    )

    print(
        f"Baseline R²: "
        f"{chrono_baseline['r2']:.3f}"
    )

    chrono_improvement = (
        chrono_baseline["mae"]
        - best_chrono_row["test_mae"]
    )

    print(
        f"\nChronological MAE improvement: "
        f"{chrono_improvement:.2f} mg/L"
    )

    if chrono_improvement > 0:
        print(
            "Result: Model improves on the chronological baseline."
        )
    else:
        print(
            "Result: Model does NOT improve on the chronological baseline."
        )

    # =========================================================================
    # RANDOM HOLDOUT
    # =========================================================================

    print("\n" + "-" * 70)
    print(
        "RANDOM HOLDOUT VALIDATION"
    )
    print("-" * 70)

    (
        X_train_random,
        X_test_random,
        y_train_random,
        y_test_random,
    ) = random_holdout(
        X,
        y,
    )

    (
        random_results,
        random_models,
        random_best,
    ) = evaluate_models(
        X_train_random,
        X_test_random,
        y_train_random,
        y_test_random,
        "random",
    )

    print(
        random_results.to_string(
            index=False
        )
    )

    print(
        f"\nBest random model by "
        f"TimeSeriesSplit CV MAE: {random_best}"
    )

    best_random_row = random_results[
        random_results["model"] == random_best
    ].iloc[0]

    print(
        f"Random holdout MAE:  "
        f"{best_random_row['test_mae']:.2f} mg/L"
    )

    print(
        f"Random holdout RMSE: "
        f"{best_random_row['test_rmse']:.2f} mg/L"
    )

    print(
        f"Random holdout R²:   "
        f"{best_random_row['test_r2']:.3f}"
    )

    # ------------------------------------------------------------------------
    # Random baseline
    # ------------------------------------------------------------------------

    random_baseline = evaluate_mean_baseline(
        y_train_random,
        y_test_random,
    )

    print(
        "\nRandom holdout mean baseline:"
    )

    print(
        f"Mean prediction: "
        f"{random_baseline['mean_prediction']:.2f} mg/L"
    )

    print(
        f"Baseline MAE: "
        f"{random_baseline['mae']:.2f} mg/L"
    )

    print(
        f"Baseline RMSE: "
        f"{random_baseline['rmse']:.2f} mg/L"
    )

    print(
        f"Baseline R²: "
        f"{random_baseline['r2']:.3f}"
    )

    random_improvement = (
        random_baseline["mae"]
        - best_random_row["test_mae"]
    )

    print(
        f"\nRandom holdout MAE improvement: "
        f"{random_improvement:.2f} mg/L"
    )

    if random_improvement > 0:
        print(
            "Result: Model improves on the random baseline."
        )
    else:
        print(
            "Result: Model does NOT improve on the random baseline."
        )

    # =========================================================================
    # ABLATION STUDY
    # =========================================================================

    print("\n" + "-" * 70)
    print(
        "FEATURE ABLATION STUDY"
    )
    print("-" * 70)

    ablation_results = run_ablation_study(
        df,
        y,
    )

    print(
        ablation_results.to_string(
            index=False
        )
    )

    best_ablation = (
        ablation_results
        .sort_values("cv_mae_mean")
        .iloc[0]
    )

    print(
        "\nBest feature configuration:"
    )

    print(
        f"Experiment: "
        f"{best_ablation['experiment']}"
    )

    print(
        f"Model: "
        f"{best_ablation['model']}"
    )

    print(
        f"TimeSeriesSplit CV MAE: "
        f"{best_ablation['cv_mae_mean']:.3f} mg/L"
    )

    # =========================================================================
    # FEATURE IMPORTANCE
    # =========================================================================

    print("\n" + "-" * 70)
    print(
        "FEATURE IMPORTANCE"
    )
    print("-" * 70)

    best_model = chrono_models[
        chrono_best
    ]

    feature_importance = extract_feature_importance(
        best_model,
        ALL_FEATURES,
    )

    print(
        feature_importance.to_string(
            index=False
        )
    )

    # =========================================================================
    # SAVE CHRONOLOGICAL PREDICTIONS
    # =========================================================================

    chronological_model = chrono_models[
        chrono_best
    ]

    chronological_predictions = chronological_model.predict(
        X_test_chrono
    )

    chronological_prediction_df = X_test_chrono.copy()

    chronological_prediction_df[
        "actual_effluent_bod5"
    ] = y_test_chrono.to_numpy()

    chronological_prediction_df[
        "predicted_effluent_bod5"
    ] = chronological_predictions

    chronological_prediction_df[
        "absolute_error"
    ] = np.abs(
        chronological_prediction_df[
            "actual_effluent_bod5"
        ]
        - chronological_prediction_df[
            "predicted_effluent_bod5"
        ]
    )

    chronological_prediction_df[
        "signed_error"
    ] = (
        chronological_prediction_df[
            "actual_effluent_bod5"
        ]
        - chronological_prediction_df[
            "predicted_effluent_bod5"
        ]
    )

    chronological_prediction_df.to_csv(
        MODELS_DIR
        / "v24_chronological_predictions.csv",
        index=False,
    )

    # =========================================================================
    # SAVE RANDOM PREDICTIONS
    # =========================================================================

    random_model = random_models[
        random_best
    ]

    random_predictions = random_model.predict(
        X_test_random
    )

    random_prediction_df = X_test_random.copy()

    random_prediction_df[
        "actual_effluent_bod5"
    ] = y_test_random.to_numpy()

    random_prediction_df[
        "predicted_effluent_bod5"
    ] = random_predictions

    random_prediction_df[
        "absolute_error"
    ] = np.abs(
        random_prediction_df[
            "actual_effluent_bod5"
        ]
        - random_prediction_df[
            "predicted_effluent_bod5"
        ]
    )

    random_prediction_df[
        "signed_error"
    ] = (
        random_prediction_df[
            "actual_effluent_bod5"
        ]
        - random_prediction_df[
            "predicted_effluent_bod5"
        ]
    )

    random_prediction_df.to_csv(
        MODELS_DIR
        / "v24_random_predictions.csv",
        index=False,
    )

    # =========================================================================
    # SAVE MODEL COMPARISON
    # =========================================================================

    model_comparison = pd.concat(
        [
            random_results,
            chrono_results,
        ],
        ignore_index=True,
    )

    model_comparison.to_csv(
        MODELS_DIR
        / "v24_model_comparison.csv",
        index=False,
    )

    # =========================================================================
    # SAVE ABLATION RESULTS
    # =========================================================================

    ablation_results.to_csv(
        MODELS_DIR
        / "v24_feature_ablation.csv",
        index=False,
    )

    # =========================================================================
    # SAVE FEATURE IMPORTANCE
    # =========================================================================

    feature_importance.to_csv(
        MODELS_DIR
        / "v24_feature_importance.csv",
        index=False,
    )

    # =========================================================================
    # SAVE FEATURE SCHEMA
    # =========================================================================

    schema_path = (
        MODELS_DIR
        / "v24_feature_schema.txt"
    )

    with open(
        schema_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "WASTEWATER AI V2.4 FEATURE SCHEMA\n"
        )

        file.write(
            "=" * 50 + "\n\n"
        )

        file.write(
            "TARGET\n"
        )

        file.write(
            "effluent_bod5\n\n"
        )

        file.write(
            "RAW INFLUENT FEATURES\n"
        )

        for feature in RAW_FEATURES:
            file.write(
                f"{feature}\n"
            )

        file.write(
            "\nENGINEERED FEATURES\n"
        )

        for feature in ENGINEERED_FEATURES:
            file.write(
                f"{feature}\n"
            )

        file.write(
            "\nTOTAL FEATURES\n"
        )

        file.write(
            f"{len(ALL_FEATURES)}\n"
        )

        file.write(
            "\nVALIDATION\n"
        )

        file.write(
            "Chronological 80/20 holdout\n"
        )

        file.write(
            "TimeSeriesSplit cross-validation\n"
        )

        file.write(
            "\nIMPORTANT\n"
        )

        file.write(
            "No downstream/process/effluent variables "
            "are used as predictors.\n"
        )

    # =========================================================================
    # FINAL DEPLOYMENT CANDIDATE
    # =========================================================================

    print("\n" + "-" * 70)
    print(
        "FINAL DEPLOYMENT CANDIDATE"
    )
    print("-" * 70)

    print(
        f"Selected model: {chrono_best}"
    )

    print(
        "Selection basis: lowest "
        "TimeSeriesSplit CV MAE"
    )

    print(
        "Refitting selected model on all valid observations..."
    )

    final_models = build_models()

    final_model = final_models[
        chrono_best
    ]

    final_model.fit(
        X,
        y,
    )

    final_model_path = (
        MODELS_DIR
        / "wastewater_bod5_model_v24.joblib"
    )

    joblib.dump(
        final_model,
        final_model_path,
    )

    # =========================================================================
    # SAVE SUMMARY
    # =========================================================================

    summary_path = (
        MODELS_DIR
        / "v24_validation_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8",
    ) as file:

        file.write(
            "WASTEWATER AI V2.4 VALIDATION SUMMARY\n"
        )

        file.write(
            "=" * 55 + "\n\n"
        )

        file.write(
            f"Raw observations: {before_drop}\n"
        )

        file.write(
            f"Complete observations: {after_drop}\n"
        )

        file.write(
            f"Final valid observations: {len(df)}\n"
        )

        file.write(
            f"Features: {len(ALL_FEATURES)}\n\n"
        )

        file.write(
            "CHRONOLOGICAL HOLDOUT\n"
        )

        file.write(
            f"Best model: {chrono_best}\n"
        )

        file.write(
            f"MAE: {best_chrono_row['test_mae']:.4f}\n"
        )

        file.write(
            f"RMSE: {best_chrono_row['test_rmse']:.4f}\n"
        )

        file.write(
            f"R2: {best_chrono_row['test_r2']:.4f}\n"
        )

        file.write(
            f"Baseline MAE: {chrono_baseline['mae']:.4f}\n"
        )

        file.write(
            f"MAE improvement: {chrono_improvement:.4f}\n\n"
        )

        file.write(
            "RANDOM HOLDOUT\n"
        )

        file.write(
            f"Best model: {random_best}\n"
        )

        file.write(
            f"MAE: {best_random_row['test_mae']:.4f}\n"
        )

        file.write(
            f"RMSE: {best_random_row['test_rmse']:.4f}\n"
        )

        file.write(
            f"R2: {best_random_row['test_r2']:.4f}\n"
        )

        file.write(
            f"Baseline MAE: {random_baseline['mae']:.4f}\n"
        )

        file.write(
            f"MAE improvement: {random_improvement:.4f}\n\n"
        )

        file.write(
            "BEST ABLATION\n"
        )

        file.write(
            f"Experiment: {best_ablation['experiment']}\n"
        )

        file.write(
            f"Model: {best_ablation['model']}\n"
        )

        file.write(
            f"CV MAE: {best_ablation['cv_mae_mean']:.4f}\n"
        )

    # =========================================================================
    # FINAL OUTPUT
    # =========================================================================

    print("\n" + "=" * 70)
    print(
        "V2.4 FILES SAVED"
    )
    print("=" * 70)

    print(
        MODELS_DIR
        / "v24_model_comparison.csv"
    )

    print(
        MODELS_DIR
        / "v24_feature_ablation.csv"
    )

    print(
        MODELS_DIR
        / "v24_chronological_predictions.csv"
    )

    print(
        MODELS_DIR
        / "v24_random_predictions.csv"
    )

    print(
        MODELS_DIR
        / "wastewater_bod5_model_v24.joblib"
    )

    print(
        MODELS_DIR
        / "v24_feature_importance.csv"
    )

    print(
        MODELS_DIR
        / "v24_feature_schema.txt"
    )

    print(
        MODELS_DIR
        / "v24_validation_summary.txt"
    )

    print("\n" + "=" * 70)
    print(
        "V2.4 TEMPORAL VALIDATION COMPLETED SUCCESSFULLY"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()
