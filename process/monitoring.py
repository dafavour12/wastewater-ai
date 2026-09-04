from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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

DEFAULT_CONTAMINATION = 0.02

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "v25_process_anomaly_model.joblib"
)


@dataclass(frozen=True)
class ProcessAnomalyResult:
    anomaly_score: float
    anomaly_percentile: float
    is_anomaly: bool
    risk_band: str
    alert_level: str
    message: str


def validate_process_input(data: dict[str, float]) -> None:
    """Validate that all required process variables are present."""

    missing = [
        feature
        for feature in PROCESS_FEATURES
        if feature not in data
    ]

    if missing:
        raise ValueError(
            f"Missing process variables: {', '.join(missing)}"
        )

    invalid = [
        feature
        for feature in PROCESS_FEATURES
        if not np.isfinite(float(data[feature]))
    ]

    if invalid:
        raise ValueError(
            f"Invalid process values: {', '.join(invalid)}"
        )


def classify_risk_band(percentile: float) -> str:
    """Convert anomaly percentile into the calibrated research band."""

    if percentile >= 99.5:
        return "critical"

    if percentile >= 99.0:
        return "high"

    if percentile >= 97.0:
        return "elevated"

    if percentile >= 90.0:
        return "low"

    return "normal"


def classify_alert_level(percentile: float) -> str:
    """Determine whether the process state should trigger an alert."""

    if percentile >= 99.0:
        return "alert"

    if percentile >= 97.0:
        return "watch"

    return "normal"


class ProcessMonitor:
    """Reusable process anomaly detection service."""

    def __init__(
        self,
        model_path: Path | str = MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Process anomaly model not found: {self.model_path}"
            )

        artifact = joblib.load(self.model_path)

        if not isinstance(artifact, dict):
            raise ValueError(
                "Invalid process anomaly model artifact."
            )

        self.model = artifact["model"]
        self.reference_scores = np.asarray(
            artifact["reference_scores"],
            dtype=float,
        )

        self.contamination = float(
            artifact.get(
                "contamination",
                DEFAULT_CONTAMINATION,
            )
        )

    def predict(
        self,
        data: dict[str, float],
    ) -> ProcessAnomalyResult:
        """Calculate anomaly status for one process observation."""

        validate_process_input(data)

        frame = pd.DataFrame(
            [[data[feature] for feature in PROCESS_FEATURES]],
            columns=PROCESS_FEATURES,
        )

        raw_prediction = int(self.model.predict(frame)[0])

        decision_score = float(
            self.model.decision_function(frame)[0]
        )

        # Lower Isolation Forest decision scores represent
        # more abnormal observations.
        percentile = float(
            np.mean(
                self.reference_scores >= decision_score
            )
            * 100.0
        )

        percentile = max(
            0.0,
            min(100.0, percentile),
        )

        risk_band = classify_risk_band(percentile)
        alert_level = classify_alert_level(percentile)

        is_anomaly = raw_prediction == -1

        if risk_band == "critical":
            message = (
                "Critical process anomaly detected. "
                "Immediate process investigation is recommended."
            )
        elif risk_band == "high":
            message = (
                "High process anomaly detected. "
                "Investigate abnormal operating conditions."
            )
        elif risk_band == "elevated":
            message = (
                "Elevated process anomaly detected. "
                "Increase monitoring and review process conditions."
            )
        elif risk_band == "low":
            message = (
                "Low-level process abnormality detected. "
                "Continue monitoring process trends."
            )
        else:
            message = (
                "Process measurements are within the normal "
                "anomaly-monitoring range."
            )

        return ProcessAnomalyResult(
            anomaly_score=decision_score,
            anomaly_percentile=percentile,
            is_anomaly=is_anomaly,
            risk_band=risk_band,
            alert_level=alert_level,
            message=message,
        )
