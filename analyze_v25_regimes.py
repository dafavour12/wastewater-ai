from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent

RAW_DATA = BASE_DIR / "data" / "raw" / "water-treatment.data"

OUTPUT_COMPARISON = (
    BASE_DIR / "models" / "v25_regime_model_comparison.csv"
)
OUTPUT_PREDICTIONS = (
    BASE_DIR / "models" / "v25_regime_predictions.csv"
)
OUTPUT_SUMMARY = (
    BASE_DIR / "models" / "v25_regime_summary.txt"
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
}


REGIME_ORDER = [
    "normal",
    "elevated",
    "extreme",
]


def load_data():
    df = pd.read_csv(
        RAW_DATA,
        header=None,
        names=UCI_COLUMNS,
        na_values=["?", ""],
    )

    for column in UCI_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    rename_map = {
        source: target
        for source, target in FEATURE_MAP.items()
    }

    rename_map["DBO-S"] = "effluent_bod5"

    df = df.rename(columns=rename_map)

    return df


def add_regime_labels(df):
    """
    Create the target regime using a pandas Series rather than
    numpy.select so that NumPy 2.x does not attempt to combine
    string labels with floating-point NaN.
    """

    df = df.copy()

    df["bod5_regime"] = pd.Series(
        pd.NA,
        index=df.index,
        dtype="string",
    )

    target = df["effluent_bod5"]

    df.loc[
        target < 30,
        "bod5_regime",
    ] = "normal"

    df.loc[
        (target >= 30) & (target < 50),
        "bod5_regime",
    ] = "elevated"

    df.loc[
        target >= 50,
        "bod5_regime",
    ] = "extreme"

    return df


def add_engineered_features(df):
    df = df.copy()

    def safe_divide(numerator, denominator):
        denominator = denominator.replace(
            0,
            np.nan,
        )

        return numerator / denominator

    df["bod5_loading"] = (
        df["influent_bod5"]
        * df["flow_m3_day"]
    )

    df["cod_loading"] = (
        df["influent_cod"]
        * df["flow_m3_day"]
    )

    df["tss_loading"] = (
        df["influent_tss"]
        * df["flow_m3_day"]
    )

    df["vss_loading"] = (
        df["influent_vss"]
        * df["flow_m3_day"]
    )

    df["cod_bod5_ratio"] = safe_divide(
        df["influent_cod"],
        df["influent_bod5"],
    )

    df["tss_bod5_ratio"] = safe_divide(
        df["influent_tss"],
        df["influent_bod5"],
    )

    df["vss_tss_ratio"] = safe_divide(
        df["influent_vss"],
        df["influent_tss"],
    )

    df["sed_tss_ratio"] = safe_divide(
        df["influent_sediments"],
        df["influent_tss"],
    )

    return df


def print_regime_distribution(df):
    print("\n" + "-" * 70)
    print("BOD5 REGIME DISTRIBUTION")
    print("-" * 70)

    counts = (
        df["bod5_regime"]
        .value_counts()
        .reindex(
            REGIME_ORDER,
            fill_value=0,
        )
    )

    total = counts.sum()

    for regime, count in counts.items():
        percentage = (
            count / total * 100
            if total
            else 0
        )

        print(
            f"{regime:10s}: "
            f"{count:3d} "
            f"({percentage:5.1f}%)"
        )


def build_feature_sets():
    raw_features = [
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

    engineered_features = [
        "bod5_loading",
        "cod_loading",
        "tss_loading",
        "vss_loading",
        "cod_bod5_ratio",
        "tss_bod5_ratio",
        "vss_tss_ratio",
        "sed_tss_ratio",
    ]

    return {
        "raw": raw_features,
        "all": raw_features + engineered_features,
    }


def build_models():
    return {
        "Logistic Regression": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "scaler",
                    StandardScaler(),
                ),
                (
                    "model",
                    LogisticRegression(
                        max_iter=3000,
                        class_weight="balanced",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "Random Forest": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "Extra Trees": Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
                (
                    "model",
                    ExtraTreesClassifier(
                        n_estimators=500,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        random_state=42,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
    }


def print_confusion_matrix(y_true, y_pred):
    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=REGIME_ORDER,
    )

    matrix_df = pd.DataFrame(
        matrix,
        index=[
            "actual_normal",
            "actual_elevated",
            "actual_extreme",
        ],
        columns=[
            "pred_normal",
            "pred_elevated",
            "pred_extreme",
        ],
    )

    print(matrix_df.to_string())

    return matrix


def main():
    print("=" * 70)
    print(
        "WASTEWATER AI - V2.5 REGIME DETECTION"
    )
    print("=" * 70)

    df = load_data()

    print(
        f"\nRaw dataset observations: "
        f"{len(df)}"
    )

    df = add_regime_labels(df)

    df = add_engineered_features(df)

    df = df.dropna(
        subset=["effluent_bod5"]
    ).copy()

    df["bod5_regime"] = (
        df["bod5_regime"]
        .astype("string")
    )

    df = df.dropna(
        subset=["bod5_regime"]
    ).copy()

    print(
        f"Valid target observations: "
        f"{len(df)}"
    )

    print_regime_distribution(df)

    feature_sets = build_feature_sets()

    results = []
    prediction_rows = []

    chronological_split = int(
        len(df) * 0.80
    )

    train_df = df.iloc[
        :chronological_split
    ].copy()

    test_df = df.iloc[
        chronological_split:
    ].copy()

    print("\n" + "-" * 70)
    print("CHRONOLOGICAL HOLDOUT")
    print("-" * 70)

    print(
        f"Training observations: "
        f"{len(train_df)}"
    )

    print(
        f"Testing observations: "
        f"{len(test_df)}"
    )

    print(
        f"Training period: "
        f"{train_df.index.min()} → "
        f"{train_df.index.max()}"
    )

    print(
        f"Testing period: "
        f"{test_df.index.min()} → "
        f"{test_df.index.max()}"
    )

    print(
        "\nTest-set regime distribution:"
    )

    test_counts = (
        test_df["bod5_regime"]
        .value_counts()
        .reindex(
            REGIME_ORDER,
            fill_value=0,
        )
    )

    for regime, count in test_counts.items():
        print(
            f"  {regime}: {count}"
        )

    for feature_set_name, features in feature_sets.items():
        X_train = train_df[features]
        X_test = test_df[features]

        y_train = train_df["bod5_regime"]
        y_test = test_df["bod5_regime"]

        models = build_models()

        for model_name, model in models.items():
            model.fit(
                X_train,
                y_train,
            )

            predictions = model.predict(
                X_test
            )

            metrics = calculate_metrics(
                y_test,
                predictions,
            )

            results.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "accuracy": metrics[
                        "accuracy"
                    ],
                    "precision_macro": metrics[
                        "precision_macro"
                    ],
                    "recall_macro": metrics[
                        "recall_macro"
                    ],
                    "f1_macro": metrics[
                        "f1_macro"
                    ],
                }
            )

            print(
                f"\n{feature_set_name} | "
                f"{model_name}"
            )

            print(
                f"Accuracy: "
                f"{metrics['accuracy']:.3f}"
            )

            print(
                f"Precision: "
                f"{metrics['precision_macro']:.3f}"
            )

            print(
                f"Recall: "
                f"{metrics['recall_macro']:.3f}"
            )

            print(
                f"F1: "
                f"{metrics['f1_macro']:.3f}"
            )

            print(
                "\nConfusion matrix:"
            )

            print_confusion_matrix(
                y_test,
                predictions,
            )

            print(
                "\nClassification report:"
            )

            print(
                classification_report(
                    y_test,
                    predictions,
                    labels=REGIME_ORDER,
                    zero_division=0,
                )
            )

            for index, actual, predicted in zip(
                test_df.index,
                y_test,
                predictions,
            ):
                prediction_rows.append(
                    {
                        "index": index,
                        "feature_set": feature_set_name,
                        "model": model_name,
                        "actual_regime": str(
                            actual
                        ),
                        "predicted_regime": str(
                            predicted
                        ),
                        "actual_bod5": test_df.loc[
                            index,
                            "effluent_bod5",
                        ],
                    }
                )

    print("\n" + "-" * 70)
    print("TIME-SERIES CROSS-VALIDATION")
    print("-" * 70)

    cv = TimeSeriesSplit(
        n_splits=5
    )

    cv_results = []

    for feature_set_name, features in feature_sets.items():
        X = df[features]
        y = df["bod5_regime"]

        for model_name, model in build_models().items():
            scores = cross_val_score(
                model,
                X,
                y,
                cv=cv,
                scoring="f1_macro",
            )

            cv_results.append(
                {
                    "feature_set": feature_set_name,
                    "model": model_name,
                    "cv_f1_macro_mean": scores.mean(),
                    "cv_f1_macro_std": scores.std(),
                }
            )

            print(
                f"{feature_set_name} | "
                f"{model_name}: "
                f"F1 = "
                f"{scores.mean():.3f} "
                f"± "
                f"{scores.std():.3f}"
            )

    results_df = pd.DataFrame(
        results
    )

    cv_df = pd.DataFrame(
        cv_results
    )

    comparison = results_df.merge(
        cv_df,
        on=[
            "feature_set",
            "model",
        ],
        how="left",
    )

    comparison = comparison.sort_values(
        [
            "cv_f1_macro_mean",
            "f1_macro",
        ],
        ascending=False,
    )

    predictions_df = pd.DataFrame(
        prediction_rows
    )

    comparison.to_csv(
        OUTPUT_COMPARISON,
        index=False,
    )

    predictions_df.to_csv(
        OUTPUT_PREDICTIONS,
        index=False,
    )

    best = comparison.iloc[0]

    raw_observation_count = len(
        load_data()
    )

    summary_lines = [
        "WASTEWATER AI - V2.5 REGIME DETECTION",
        "=" * 60,
        "",
        f"Raw observations: "
        f"{raw_observation_count}",
        (
            "Valid target observations: "
            f"{len(df)}"
        ),
        "",
        "REGIME DEFINITIONS",
        "-" * 40,
        "normal: BOD5 < 30 mg/L",
        (
            "elevated: "
            "30 <= BOD5 < 50 mg/L"
        ),
        "extreme: BOD5 >= 50 mg/L",
        "",
        "REGIME COUNTS",
        "-" * 40,
    ]

    counts = (
        df["bod5_regime"]
        .value_counts()
        .reindex(
            REGIME_ORDER,
            fill_value=0,
        )
    )

    for regime, count in counts.items():
        summary_lines.append(
            f"{regime}: {count}"
        )

    summary_lines.extend(
        [
            "",
            "TEST-SET REGIME COUNTS",
            "-" * 40,
        ]
    )

    for regime, count in test_counts.items():
        summary_lines.append(
            f"{regime}: {count}"
        )

    summary_lines.extend(
        [
            "",
            "BEST MODEL BY TIME-SERIES CV F1",
            "-" * 40,
            (
                f"Feature set: "
                f"{best['feature_set']}"
            ),
            f"Model: {best['model']}",
            (
                "CV F1 macro: "
                f"{best['cv_f1_macro_mean']:.4f}"
            ),
            (
                "Holdout F1 macro: "
                f"{best['f1_macro']:.4f}"
            ),
            (
                "Holdout accuracy: "
                f"{best['accuracy']:.4f}"
            ),
            (
                "Holdout macro recall: "
                f"{best['recall_macro']:.4f}"
            ),
            "",
            "FILES",
            "-" * 40,
            str(OUTPUT_COMPARISON),
            str(OUTPUT_PREDICTIONS),
            str(OUTPUT_SUMMARY),
        ]
    )

    OUTPUT_SUMMARY.write_text(
        "\n".join(summary_lines),
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print(
        "V2.5 REGIME ANALYSIS FILES SAVED"
    )
    print("=" * 70)

    print(OUTPUT_COMPARISON)
    print(OUTPUT_PREDICTIONS)
    print(OUTPUT_SUMMARY)

    print("\n" + "=" * 70)
    print(
        "V2.5 REGIME DETECTION COMPLETED"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()