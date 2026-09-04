from __future__ import annotations

from typing import Any

from risk.scoring import (
    RiskAssessment,
    assess_risk,
)
from scenario.models import ScenarioInput


def _extract_anomaly_percentile(
    process_result: Any,
) -> float:
    """
    Extract the anomaly percentile from a process prediction result.

    The scenario layer intentionally accepts either:

    1. a mapping containing "anomaly_percentile", or
    2. an object exposing an anomaly_percentile attribute.

    This keeps the scenario risk adapter independent of the
    concrete process-monitoring implementation.
    """

    if isinstance(process_result, dict):
        if "anomaly_percentile" not in process_result:
            raise ValueError(
                "Process predictor result is missing "
                "'anomaly_percentile'."
            )

        value = process_result["anomaly_percentile"]

    else:
        if not hasattr(
            process_result,
            "anomaly_percentile",
        ):
            raise ValueError(
                "Process predictor result is missing "
                "'anomaly_percentile'."
            )

        value = process_result.anomaly_percentile

    try:
        anomaly_percentile = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Process predictor returned an invalid "
            "anomaly_percentile."
        ) from exc

    if not 0.0 <= anomaly_percentile <= 100.0:
        raise ValueError(
            "anomaly_percentile must be between 0 and 100."
        )

    return anomaly_percentile


def assess_scenario_risk(
    scenario: ScenarioInput,
    predicted_bod5: float,
    process_result: Any,
) -> RiskAssessment:
    """
    Assess the risk of one wastewater treatment scenario.

    This is an integration adapter around the existing V2.7
    risk engine.

    It does not implement new risk-scoring rules.

    Parameters
    ----------
    scenario:
        The wastewater treatment scenario being evaluated.

    predicted_bod5:
        Effluent BOD5 predicted for the scenario.

    process_result:
        Output from the process anomaly detector.

    Returns
    -------
    RiskAssessment
        The existing V2.7 risk assessment object.
    """

    if not isinstance(
        scenario,
        ScenarioInput,
    ):
        raise TypeError(
            "scenario must be a ScenarioInput."
        )

    try:
        predicted_bod5_value = float(
            predicted_bod5
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "predicted_bod5 must be numeric."
        ) from exc

    if predicted_bod5_value < 0:
        raise ValueError(
            "predicted_bod5 must not be negative."
        )

    anomaly_percentile = _extract_anomaly_percentile(
        process_result
    )

    assessment = assess_risk(
        predicted_bod5=predicted_bod5_value,
        anomaly_percentile=anomaly_percentile,
        model_confidence=scenario.model_confidence,
    )

    if not isinstance(
        assessment,
        RiskAssessment,
    ):
        raise TypeError(
            "assess_risk must return a RiskAssessment."
        )

    return assessment
