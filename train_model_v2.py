from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import (
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, train_test_split


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "wastewater_v2.csv"
MODEL_DIR = BASE_DIR / "models"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


TARGET = "effluent_bod5"

# Only variables that are available before treatment/output.
# Do NOT include effluent variables because that would leak the target.
FEATURES = [
    "flow_m3_day",
    "influent_ph",
    "influent_bod5",
    "influent_cod",
    "influent_tss",
]


def evaluate_metrics(y_true, y_pred):
    """Return standard regression metrics."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)

    return mae, rmse, r2


def main():
    print("=" * 70)
    print("WASTEWATER AI - MODEL V2 TRAINING")
    print("=" * 70)

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Processed dataset not found: {DATA_PATH}"
        )

    df = pd.read_csv(DATA_PATH)

    required_columns = FEATURES + [TARGET]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df = df[required_columns].dropna().copy()

    X = df[FEATURES]
    y = df[TARGET]

    print(f"\nDataset rows: {len(df)}")
    print(f"Features: {FEATURES}")
    print(f"Target: {TARGET}")

    # ---------------------------------------------------------
    # Holdout test set
    # ---------------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    print(f"\nTraining rows: {len(X_train)}")
    print(f"Holdout test rows: {len(X_test)}")

    # ---------------------------------------------------------
    # Candidate models
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # Cross-validation
    # ---------------------------------------------------------
    cv = KFold(
        n_splits=5,
        shuffle=True,
        random_state=42,
    )

    results = []
    test_predictions = {}

    print("\n" + "-" * 70)
    print("MODEL EVALUATION")
    print("-" * 70)

    for name, model in models.items():
        # Train only on training partition.
        model.fit(X_train, y_train)

        # Completely untouched holdout predictions.
        test_pred = model.predict(X_test)

        test_mae, test_rmse, test_r2 = evaluate_metrics(
            y_test,
            test_pred,
        )

        test_predictions[name] = test_pred

        # Cross-validation is performed ONLY on the training data.
        cv_mae = []
        cv_rmse = []
        cv_r2 = []

        for train_idx, val_idx in cv.split(X_train):
            X_cv_train = X_train.iloc[train_idx]
            X_cv_val = X_train.iloc[val_idx]

            y_cv_train = y_train.iloc[train_idx]
            y_cv_val = y_train.iloc[val_idx]

            cv_model = model.__class__(**model.get_params())
            cv_model.fit(X_cv_train, y_cv_train)

            cv_pred = cv_model.predict(X_cv_val)

            mae, rmse, r2 = evaluate_metrics(
                y_cv_val,
                cv_pred,
            )

            cv_mae.append(mae)
            cv_rmse.append(rmse)
            cv_r2.append(r2)

        results.append(
            {
                "model": name,
                "test_mae": test_mae,
                "test_rmse": test_rmse,
                "test_r2": test_r2,
                "cv_mae_mean": sum(cv_mae) / len(cv_mae),
                "cv_mae_std": pd.Series(cv_mae).std(),
                "cv_rmse_mean": sum(cv_rmse) / len(cv_rmse),
                "cv_rmse_std": pd.Series(cv_rmse).std(),
                "cv_r2_mean": sum(cv_r2) / len(cv_r2),
                "cv_r2_std": pd.Series(cv_r2).std(),
            }
        )

    results_df = pd.DataFrame(results)

    # Best model is selected using CV MAE.
    results_df = results_df.sort_values(
        by="cv_mae_mean",
        ascending=True,
    ).reset_index(drop=True)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))

    comparison_path = MODEL_DIR / "v2_model_comparison.csv"
    results_df.to_csv(comparison_path, index=False)

    best_model_name = results_df.iloc[0]["model"]

    print("\n" + "=" * 70)
    print(f"BEST MODEL: {best_model_name}")
    print("=" * 70)

    best_test_pred = test_predictions[best_model_name]

    best_test_mae, best_test_rmse, best_test_r2 = evaluate_metrics(
        y_test,
        best_test_pred,
    )

    print(f"Holdout MAE:  {best_test_mae:.2f} mg/L")
    print(f"Holdout RMSE: {best_test_rmse:.2f} mg/L")
    print(f"Holdout R²:   {best_test_r2:.3f}")

    # ---------------------------------------------------------
    # Save untouched holdout predictions
    # ---------------------------------------------------------
    holdout_predictions = X_test.copy()
    holdout_predictions["actual_effluent_bod5"] = y_test.values
    holdout_predictions["predicted_effluent_bod5"] = best_test_pred
    holdout_predictions["absolute_error"] = (
        holdout_predictions["actual_effluent_bod5"]
        - holdout_predictions["predicted_effluent_bod5"]
    ).abs()

    holdout_predictions = holdout_predictions.sort_index()

    holdout_path = MODEL_DIR / "v2_test_predictions.csv"
    holdout_predictions.to_csv(
        holdout_path,
        index=False,
    )

    # ---------------------------------------------------------
    # Refit selected model on ALL available data for deployment
    # ---------------------------------------------------------
    final_model = models[best_model_name]
    final_model.fit(X, y)

    model_path = MODEL_DIR / "wastewater_bod5_model_v2.joblib"
    joblib.dump(final_model, model_path)

    # ---------------------------------------------------------
    # Feature importance
    # ---------------------------------------------------------
    if hasattr(final_model, "feature_importances_"):
        importance_df = pd.DataFrame(
            {
                "feature": FEATURES,
                "importance": final_model.feature_importances_,
            }
        ).sort_values(
            by="importance",
            ascending=False,
        )

        importance_path = MODEL_DIR / "v2_feature_importance.csv"
        importance_df.to_csv(
            importance_path,
            index=False,
        )

        print("\nFeature importance:")
        print(importance_df.to_string(index=False))

    # ---------------------------------------------------------
    # Example prediction
    # ---------------------------------------------------------
    sample = pd.DataFrame(
        [
            {
                "flow_m3_day": 37000,
                "influent_ph": 7.8,
                "influent_bod5": 190,
                "influent_cod": 405,
                "influent_tss": 230,
            }
        ]
    )

    sample_prediction = final_model.predict(sample)[0]

    print("\nExample prediction:")
    print(f"Predicted effluent BOD5: {sample_prediction:.2f} mg/L")

    print("\nSaved files:")
    print(f"- {comparison_path}")
    print(f"- {holdout_path}")
    print(f"- {model_path}")

    if hasattr(final_model, "feature_importances_"):
        print(f"- {importance_path}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()
