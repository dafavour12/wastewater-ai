from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "raw" / "water-treatment.data"

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "v25_process_anomaly_model.joblib"
)


COLUMNS = [
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


CONTAMINATION = 0.02


def main() -> None:
    print("Loading UCI wastewater dataset...")

    df = pd.read_csv(
        DATA_PATH,
        header=None,
        names=COLUMNS,
        na_values=["?"],
    )

    process_df = df[PROCESS_FEATURES].copy()

    print(f"Raw observations: {len(df)}")

    process_df = process_df.apply(
        pd.to_numeric,
        errors="coerce",
    )

    process_df = process_df.dropna()

    print(
        f"Complete process observations: "
        f"{len(process_df)}"
    )

    model = IsolationForest(
        n_estimators=400,
        contamination=CONTAMINATION,
        random_state=42,
    )

    model.fit(process_df)

    reference_scores = model.decision_function(
        process_df
    )

    artifact = {
        "model": model,
        "reference_scores": reference_scores,
        "features": PROCESS_FEATURES,
        "contamination": CONTAMINATION,
        "training_rows": len(process_df),
        "description": (
            "Wastewater AI V2.5 process anomaly "
            "monitoring model."
        ),
    }

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        artifact,
        MODEL_PATH,
    )

    print()
    print("Process monitoring model created.")
    print(f"Model: {MODEL_PATH}")
    print(f"Features: {len(PROCESS_FEATURES)}")
    print(f"Contamination: {CONTAMINATION:.0%}")


if __name__ == "__main__":
    main()
