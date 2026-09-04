from dataclasses import dataclass
from enum import IntEnum


class RiskLevel(IntEnum):
    """
    Ordered operational risk levels.

    The numeric values allow straightforward comparison:
    NORMAL < LOW < ELEVATED < HIGH < CRITICAL.
    """

    NORMAL = 0
    LOW = 1
    ELEVATED = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class RiskAssessment:
    """
    Final risk assessment returned by the risk engine.
    """

    risk_level: RiskLevel
    risk_score: float
    prediction_score: float
    anomaly_score: float
    confidence_score: float
    predicted_bod5: float
    anomaly_percentile: float
    risk_reason: str
    recommended_action: str


def clamp(value: float, minimum: float, maximum: float) -> float:
    """
    Restrict a numeric value to a defined range.
    """
    return max(minimum, min(value, maximum))


def calculate_prediction_score(
    predicted_bod5: float,
) -> float:
    """
    Convert predicted effluent BOD5 into a 0-100 risk score.

    Candidate engineering bands:

    <=20       NORMAL
    21-30      LOW
    31-50      ELEVATED
    51-75      HIGH
    >75        CRITICAL
    """

    if predicted_bod5 < 0:
        raise ValueError(
            "predicted_bod5 must be non-negative."
        )

    if predicted_bod5 <= 20:
        return 0.0

    if predicted_bod5 <= 30:
        return 25.0

    if predicted_bod5 <= 50:
        return 50.0

    if predicted_bod5 <= 75:
        return 75.0

    # Values above 75 continue increasing but are capped
    # at 100 so the score remains bounded.
    score = 75.0 + (
        (predicted_bod5 - 75.0)
        / 75.0
        * 25.0
    )

    return clamp(
        score,
        0.0,
        100.0,
    )


def calculate_anomaly_score(
    anomaly_percentile: float,
) -> float:
    """
    Convert anomaly percentile directly into a 0-100
    anomaly risk score.

    A higher percentile means a more anomalous process state.
    """

    if not 0 <= anomaly_percentile <= 100:
        raise ValueError(
            "anomaly_percentile must be between 0 and 100."
        )

    return float(anomaly_percentile)


def calculate_confidence_score(
    model_confidence: str,
) -> float:
    """
    Convert qualitative model confidence into a bounded score.

    This is intentionally conservative because the current
    V2.4 model has not undergone production-grade uncertainty
    calibration.
    """

    normalized = model_confidence.strip().lower()

    mapping = {
        "high": 100.0,
        "medium": 70.0,
        "moderate": 70.0,
        "low": 40.0,
        "limited": 25.0,
        "research": 25.0,
    }

    if normalized not in mapping:
        raise ValueError(
            "model_confidence must be one of: "
            "high, medium, moderate, low, limited, research."
        )

    return mapping[normalized]


def level_from_prediction(
    predicted_bod5: float,
) -> RiskLevel:
    """
    Determine the risk level contributed by predicted BOD5.
    """

    if predicted_bod5 <= 20:
        return RiskLevel.NORMAL

    if predicted_bod5 <= 30:
        return RiskLevel.LOW

    if predicted_bod5 <= 50:
        return RiskLevel.ELEVATED

    if predicted_bod5 <= 75:
        return RiskLevel.HIGH

    return RiskLevel.CRITICAL


def level_from_anomaly(
    anomaly_percentile: float,
) -> RiskLevel:
    """
    Determine the candidate risk level contributed by
    process anomaly percentile.

    Candidate calibration:

    <90       NORMAL
    90-97     LOW
    97-99     ELEVATED
    99-99.5   HIGH
    >=99.5    CRITICAL
    """

    if not 0 <= anomaly_percentile <= 100:
        raise ValueError(
            "anomaly_percentile must be between 0 and 100."
        )

    if anomaly_percentile < 90:
        return RiskLevel.NORMAL

    if anomaly_percentile < 97:
        return RiskLevel.LOW

    if anomaly_percentile < 99:
        return RiskLevel.ELEVATED

    if anomaly_percentile < 99.5:
        return RiskLevel.HIGH

    return RiskLevel.CRITICAL


def determine_overall_level(
    predicted_bod5: float,
    anomaly_percentile: float,
) -> RiskLevel:
    """
    Combine prediction and anomaly risk using the higher
    contributing risk level plus explicit safety overrides.

    Safety rules:

    1. Critical anomaly => CRITICAL.
    2. High anomaly + elevated-or-higher BOD5 prediction
       => CRITICAL.
    3. Otherwise use the higher of prediction/anomaly levels.
    """

    prediction_level = level_from_prediction(
        predicted_bod5
    )

    anomaly_level = level_from_anomaly(
        anomaly_percentile
    )

    # Critical process anomaly is never hidden by a
    # reassuring BOD5 prediction.
    if anomaly_percentile >= 99.5:
        return RiskLevel.CRITICAL

    # A high anomaly combined with elevated predicted BOD5
    # represents a compounded risk.
    if (
        anomaly_percentile >= 99.0
        and predicted_bod5 > 30
    ):
        return RiskLevel.CRITICAL

    return max(
        prediction_level,
        anomaly_level,
    )


def build_risk_reason(
    predicted_bod5: float,
    anomaly_percentile: float,
    prediction_level: RiskLevel,
    anomaly_level: RiskLevel,
    overall_level: RiskLevel,
) -> str:
    """
    Generate a concise human-readable explanation.
    """

    if anomaly_percentile >= 99.5:
        return (
            "Critical process anomaly detected "
            f"(anomaly percentile "
            f"{anomaly_percentile:.1f}). "
            "The anomaly safety override sets the "
            "overall risk to CRITICAL."
        )

    if (
        anomaly_percentile >= 99.0
        and predicted_bod5 > 30
    ):
        return (
            "Elevated predicted effluent BOD5 combined "
            "with a high process anomaly signal. "
            "The combined-risk safety rule sets the "
            "overall risk to CRITICAL."
        )

    if (
        prediction_level == RiskLevel.NORMAL
        and anomaly_level == RiskLevel.NORMAL
    ):
        return (
            "Predicted effluent BOD5 and process anomaly "
            "signals are both within the normal candidate range."
        )

    if prediction_level >= anomaly_level:
        return (
            "Predicted effluent BOD5 is the dominant "
            "risk signal."
        )

    if anomaly_level > prediction_level:
        return (
            "Process anomaly is the dominant risk signal "
            "despite the lower BOD5 prediction."
        )

    return (
        f"Combined prediction and process signals indicate "
        f"{overall_level.name} risk."
    )


def build_recommended_action(
    risk_level: RiskLevel,
) -> str:
    """
    Convert risk level into a practical candidate action.

    These are operational guidance categories, not regulatory
    instructions.
    """

    actions = {
        RiskLevel.NORMAL: (
            "Continue routine monitoring."
        ),
        RiskLevel.LOW: (
            "Continue monitoring and review recent process trends."
        ),
        RiskLevel.ELEVATED: (
            "Increase monitoring frequency and inspect relevant "
            "process conditions."
        ),
        RiskLevel.HIGH: (
            "Prioritize process review and investigate abnormal "
            "operating conditions."
        ),
        RiskLevel.CRITICAL: (
            "Immediate process investigation recommended; "
            "verify measurements and inspect treatment performance."
        ),
    }

    return actions[risk_level]


def assess_risk(
    predicted_bod5: float,
    anomaly_percentile: float,
    model_confidence: str = "research",
) -> RiskAssessment:
    """
    Produce a complete Wastewater AI risk assessment.

    Parameters
    ----------
    predicted_bod5:
        Predicted effluent BOD5 in mg/L.

    anomaly_percentile:
        Process anomaly percentile from 0 to 100.

    model_confidence:
        Qualitative model-confidence category.

    Returns
    -------
    RiskAssessment
        Transparent risk score and explanation.
    """

    if predicted_bod5 < 0:
        raise ValueError(
            "predicted_bod5 must be non-negative."
        )

    prediction_score = (
        calculate_prediction_score(
            predicted_bod5
        )
    )

    anomaly_score = (
        calculate_anomaly_score(
            anomaly_percentile
        )
    )

    confidence_score = (
        calculate_confidence_score(
            model_confidence
        )
    )

    # Confidence has a small weighting because the current
    # model does not yet have rigorous uncertainty calibration.
    risk_score = (
        0.50 * prediction_score
        + 0.40 * anomaly_score
        + 0.10 * confidence_score
    )

    risk_score = clamp(
        risk_score,
        0.0,
        100.0,
    )

    prediction_level = (
        level_from_prediction(
            predicted_bod5
        )
    )

    anomaly_level = (
        level_from_anomaly(
            anomaly_percentile
        )
    )

    overall_level = determine_overall_level(
        predicted_bod5,
        anomaly_percentile,
    )

    # Safety overrides must also affect the numerical score.
    if overall_level == RiskLevel.CRITICAL:
        if anomaly_percentile >= 99.5:
            risk_score = max(
                risk_score,
                90.0,
            )
        elif (
            anomaly_percentile >= 99
            and predicted_bod5 > 30
        ):
            risk_score = max(
                risk_score,
                85.0,
            )

    reason = build_risk_reason(
        predicted_bod5=predicted_bod5,
        anomaly_percentile=anomaly_percentile,
        prediction_level=prediction_level,
        anomaly_level=anomaly_level,
        overall_level=overall_level,
    )

    action = build_recommended_action(
        overall_level
    )

    return RiskAssessment(
        risk_level=overall_level,
        risk_score=round(
            risk_score,
            2,
        ),
        prediction_score=round(
            prediction_score,
            2,
        ),
        anomaly_score=round(
            anomaly_score,
            2,
        ),
        confidence_score=round(
            confidence_score,
            2,
        ),
        predicted_bod5=float(
            predicted_bod5
        ),
        anomaly_percentile=float(
            anomaly_percentile
        ),
        risk_reason=reason,
        recommended_action=action,
    )
