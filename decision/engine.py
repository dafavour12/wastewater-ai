from __future__ import annotations

import math

from decision.models import (
    DecisionInput,
    DecisionPriority,
    DecisionRecommendation,
)
from decision.rules import (
    BOD5_HIGH_THRESHOLD,
    BOD5_MODERATE_THRESHOLD,
    BOD5_VERY_HIGH_THRESHOLD,
    CRITICAL_ANOMALY_PERCENTILE,
    FLOW_INVESTIGATION_THRESHOLD,
    HIGH_ANOMALY_PERCENTILE,
    LOW_DO_THRESHOLD,
    LOW_HRT_THRESHOLD,
)


def _validate_input(data: DecisionInput) -> None:
    numeric_values = {
        "predicted_effluent_bod5": data.predicted_effluent_bod5,
        "anomaly_percentile": data.anomaly_percentile,
        "overall_risk_score": data.overall_risk_score,
        "dissolved_oxygen": data.dissolved_oxygen,
        "flow_m3_day": data.flow_m3_day,
        "hrt_hours": data.hrt_hours,
        "influent_bod5": data.influent_bod5,
        "influent_cod": data.influent_cod,
        "influent_tss": data.influent_tss,
    }

    for field_name, value in numeric_values.items():
        if not math.isfinite(value):
            raise ValueError(
                f"{field_name} must be a finite number."
            )

    if data.predicted_effluent_bod5 < 0:
        raise ValueError(
            "predicted_effluent_bod5 cannot be negative."
        )

    if not 0 <= data.anomaly_percentile <= 100:
        raise ValueError(
            "anomaly_percentile must be between 0 and 100."
        )

    if data.overall_risk_level < 0:
        raise ValueError(
            "overall_risk_level cannot be negative."
        )

    if data.overall_risk_score < 0:
        raise ValueError(
            "overall_risk_score cannot be negative."
        )

    if data.dissolved_oxygen < 0:
        raise ValueError(
            "dissolved_oxygen cannot be negative."
        )

    if data.flow_m3_day <= 0:
        raise ValueError(
            "flow_m3_day must be greater than zero."
        )

    if data.hrt_hours <= 0:
        raise ValueError(
            "hrt_hours must be greater than zero."
        )

    if data.influent_bod5 < 0:
        raise ValueError(
            "influent_bod5 cannot be negative."
        )

    if data.influent_cod < 0:
        raise ValueError(
            "influent_cod cannot be negative."
        )

    if data.influent_tss < 0:
        raise ValueError(
            "influent_tss cannot be negative."
        )


def _add_unique(
    items: list[str],
    values: list[str],
) -> None:
    for value in values:
        if value not in items:
            items.append(value)


def generate_decision(
    data: DecisionInput,
) -> DecisionRecommendation:
    """
    Generate deterministic engineering investigation recommendations.

    The engine does not diagnose a root cause.
    It identifies possible contributors and recommends checks
    based on the available indicators.
    """

    _validate_input(data)

    priority = DecisionPriority.NORMAL

    possible_contributors: list[str] = []
    checks_to_perform: list[str] = []
    recommended_actions: list[str] = []
    monitoring_recommendations: list[str] = []
    evidence: list[str] = []
    limitations: list[str] = []

    # ---------------------------------------------------------
    # BOD5
    # ---------------------------------------------------------

    if data.predicted_effluent_bod5 > BOD5_VERY_HIGH_THRESHOLD:

        priority = max(
            priority,
            DecisionPriority.CRITICAL,
        )

        _add_unique(
            possible_contributors,
            [
                "Very high predicted effluent BOD5",
            ],
        )

        _add_unique(
            checks_to_perform,
            [
                "Confirm the predicted effluent BOD5 with laboratory testing.",
                "Review recent influent organic loading.",
                "Review biological treatment performance.",
            ],
        )

        _add_unique(
            recommended_actions,
            [
                "Prioritize an immediate process-performance investigation.",
            ],
        )

        _add_unique(
            monitoring_recommendations,
            [
                "Increase short-term monitoring of influent and effluent BOD5.",
            ],
        )

        _add_unique(
            evidence,
            [
                "Predicted effluent BOD5 is above the very-high "
                "investigation threshold.",
            ],
        )

    elif data.predicted_effluent_bod5 > BOD5_HIGH_THRESHOLD:

        priority = max(
            priority,
            DecisionPriority.HIGH,
        )

        _add_unique(
            possible_contributors,
            [
                "High predicted effluent BOD5",
            ],
        )

        _add_unique(
            checks_to_perform,
            [
                "Review recent influent organic loading.",
                "Review aeration and biological treatment performance.",
                "Confirm the effluent BOD5 result with laboratory testing.",
            ],
        )

        _add_unique(
            recommended_actions,
            [
                "Prioritize a treatment-process performance review.",
            ],
        )

        _add_unique(
            monitoring_recommendations,
            [
                "Increase monitoring of BOD5 and relevant process conditions.",
            ],
        )

        _add_unique(
            evidence,
            [
                "Predicted effluent BOD5 is above the high-performance "
                "investigation threshold.",
            ],
        )

    elif data.predicted_effluent_bod5 > BOD5_MODERATE_THRESHOLD:

        priority = max(
            priority,
            DecisionPriority.MODERATE,
        )

        _add_unique(
            possible_contributors,
            [
                "Moderately elevated predicted effluent BOD5",
            ],
        )

        _add_unique(
            checks_to_perform,
            [
                "Review recent influent loading.",
                "Review recent process trends.",
            ],
        )

        _add_unique(
            recommended_actions,
            [
                "Continue monitoring and investigate persistent deterioration.",
            ],
        )

        _add_unique(
            monitoring_recommendations,
            [
                "Maintain routine BOD5 and process monitoring.",
            ],
        )

        _add_unique(
            evidence,
            [
                "Predicted effluent BOD5 is above the normal "
                "investigation range.",
            ],
        )

    # ---------------------------------------------------------
    # Dissolved oxygen
    # ---------------------------------------------------------

    if data.dissolved_oxygen < LOW_DO_THRESHOLD:

        priority = max(
            priority,
            DecisionPriority.HIGH,
        )

        _add_unique(
            possible_contributors,
            [
                "Low dissolved oxygen",
            ],
        )

        _add_unique(
            checks_to_perform,
            [
                "Verify dissolved oxygen sensor calibration and measurement quality.",
                "Check aeration/blower operation.",
                "Review air supply and mixing conditions.",
                "Review changes in influent loading and flow.",
            ],
        )

        _add_unique(
            recommended_actions,
            [
                "Investigate aeration performance before making major "
                "process changes.",
            ],
        )

        _add_unique(
            monitoring_recommendations,
            [
                "Increase dissolved oxygen monitoring frequency.",
            ],
        )

        _add_unique(
            evidence,
            [
                "Measured dissolved oxygen is below the initial "
                "low-DO investigation threshold.",
            ],
        )

    # ---------------------------------------------------------
    # Flow
    # ---------------------------------------------------------

    if data.flow_m3_day > FLOW_INVESTIGATION_THRESHOLD:

        priority = max(
            priority,
            DecisionPriority.HIGH,
        )

        _add_unique(
            possible_contributors,
            [
                "Elevated hydraulic loading",
            ],
        )

        _add_unique(
            checks_to_perform,
            [
                "Review current flow against the plant's normal/design flow.",
                "Check for unusual peak-flow events.",
                "Review hydraulic retention time.",
            ],
        )

        _add_unique(
            recommended_actions,
            [
                "Investigate whether hydraulic loading is affecting "
                "treatment performance.",
            ],
        )

        _add_unique(
            monitoring_recommendations,
            [
                "Increase flow monitoring during peak periods.",
            ],
        )

        _add_unique(
            evidence,
            [
                "Reported flow exceeds the configured hydraulic "
                "investigation threshold.",
            ],
        )

    # ---------------------------------------------------------
    # Hydraulic retention time
    # ---------------------------------------------------------

    if data.hrt_hours < LOW_HRT_THRESHOLD:

        priority = max(
            priority,
            DecisionPriority.HIGH,
        )

        _add_unique(
            possible_contributors,
            [
                "Low hydraulic retention time",
            ],
        )

        _add_unique(
            checks_to_perform,
            [
                "Review hydraulic loading and tank operating conditions.",
                "Check whether treatment units are hydraulically overloaded.",
            ],
        )

        _add_unique(
            recommended_actions,
            [
                "Investigate hydraulic performance and retention time.",
            ],
        )

        _add_unique(
            monitoring_recommendations,
            [
                "Monitor flow and hydraulic retention time during "
                "peak loading.",
            ],
        )

        _add_unique(
            evidence,
            [
                "Reported hydraulic retention time is below the "
                "investigation threshold.",
            ],
        )

    # ---------------------------------------------------------
    # Process anomaly
    # ---------------------------------------------------------

    if data.anomaly_percentile >= CRITICAL_ANOMALY_PERCENTILE:

        priority = max(
            priority,
            DecisionPriority.CRITICAL,
        )

        _add_unique(
            possible_contributors,
            [
                "Critical process anomaly score",
            ],
        )

        _add_unique(
            checks_to_perform,
            [
                "Immediately review the process measurements "
                "contributing to the anomaly.",
                "Verify critical sensors and instrumentation.",
                "Review recent operational events and process changes.",
            ],
        )

        _add_unique(
            recommended_actions,
            [
                "Escalate the condition for immediate "
                "engineering/operator review.",
            ],
        )

        _add_unique(
            monitoring_recommendations,
            [
                "Use increased monitoring until the abnormal condition "
                "is resolved or explained.",
            ],
        )

        _add_unique(
            evidence,
            [
                "The process anomaly percentile is within the "
                "critical investigation range.",
            ],
        )

    elif data.anomaly_percentile >= HIGH_ANOMALY_PERCENTILE:

        priority = max(
            priority,
            DecisionPriority.HIGH,
        )

        _add_unique(
            possible_contributors,
            [
                "High process anomaly score",
            ],
        )

        _add_unique(
            checks_to_perform,
            [
                "Review the process measurements that produced the anomaly.",
                "Check sensors and instrumentation for abnormal readings.",
                "Compare current process conditions with recent historical values.",
            ],
        )

        _add_unique(
            recommended_actions,
            [
                "Prioritize process and instrumentation investigation.",
            ],
        )

        _add_unique(
            monitoring_recommendations,
            [
                "Increase monitoring until the abnormal condition "
                "is explained.",
            ],
        )

        _add_unique(
            evidence,
            [
                "The process anomaly percentile indicates an unusually "
                "abnormal operating condition.",
            ],
        )

    # ---------------------------------------------------------
    # Model confidence
    # ---------------------------------------------------------

    confidence = data.model_confidence.strip().lower()

    if confidence in {
        "low",
        "limited",
        "research",
        "unknown",
    }:

        limitations.append(
            "Model confidence is limited; verify important decisions "
            "with laboratory measurements and engineering judgement."
        )

    # ---------------------------------------------------------
    # Overall risk escalation
    # ---------------------------------------------------------

    if data.overall_risk_level >= 4:

        priority = max(
            priority,
            DecisionPriority.CRITICAL,
        )

        _add_unique(
            recommended_actions,
            [
                "Escalate the assessment for immediate "
                "engineering/operator review.",
            ],
        )

        _add_unique(
            evidence,
            [
                "The combined risk assessment is at a critical level.",
            ],
        )

    elif data.overall_risk_level >= 3:

        priority = max(
            priority,
            DecisionPriority.HIGH,
        )

        _add_unique(
            recommended_actions,
            [
                "Prioritize review of the contributing process conditions.",
            ],
        )

        _add_unique(
            evidence,
            [
                "The combined risk assessment is at a high level.",
            ],
        )

    # ---------------------------------------------------------
    # Normal condition
    # ---------------------------------------------------------

    if priority == DecisionPriority.NORMAL:

        summary = (
            "No major investigation trigger was identified. "
            "Continue routine process monitoring."
        )

        checks_to_perform.append(
            "Continue routine verification of process measurements."
        )

        recommended_actions.append(
            "Maintain normal monitoring and operational procedures."
        )

        monitoring_recommendations.append(
            "Continue routine monitoring of treatment performance."
        )

        evidence.append(
            "Current prediction, anomaly, risk, and operating-condition "
            "inputs did not trigger an elevated investigation rule."
        )

    else:

        summary = (
            f"{priority.name.title()} priority investigation recommended "
            "based on the available prediction, process, and risk indicators."
        )

    return DecisionRecommendation(
        priority=priority,
        summary=summary,
        possible_contributors=possible_contributors,
        checks_to_perform=checks_to_perform,
        recommended_actions=recommended_actions,
        monitoring_recommendations=monitoring_recommendations,
        evidence=evidence,
        limitations=limitations,
    )
