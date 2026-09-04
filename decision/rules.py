from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecisionRule:
    name: str
    priority: int
    contributor: str
    checks: tuple[str, ...]
    actions: tuple[str, ...]
    monitoring: tuple[str, ...]
    evidence_template: str


# These thresholds are intentionally conservative starting points.
# They should become configurable as the project matures.
RULES = (
    DecisionRule(
        name="very_high_bod5",
        priority=4,
        contributor="Very high predicted effluent BOD5",
        checks=(
            "Confirm the predicted effluent BOD5 with laboratory testing.",
            "Review recent influent organic loading.",
            "Review biological treatment performance.",
        ),
        actions=(
            "Prioritize an immediate process-performance investigation.",
        ),
        monitoring=(
            "Increase short-term monitoring of influent and effluent BOD5.",
        ),
        evidence_template=(
            "Predicted effluent BOD5 is above the very-high performance "
            "threshold."
        ),
    ),
    DecisionRule(
        name="high_bod5",
        priority=3,
        contributor="High predicted effluent BOD5",
        checks=(
            "Review recent influent organic loading.",
            "Review aeration and biological treatment performance.",
            "Confirm the effluent BOD5 result with laboratory testing.",
        ),
        actions=(
            "Prioritize a treatment-process performance review.",
        ),
        monitoring=(
            "Increase monitoring of BOD5 and relevant process conditions.",
        ),
        evidence_template=(
            "Predicted effluent BOD5 is above the high-performance threshold."
        ),
    ),
    DecisionRule(
        name="moderate_bod5",
        priority=2,
        contributor="Moderately elevated predicted effluent BOD5",
        checks=(
            "Review recent influent loading.",
            "Review recent process trends.",
        ),
        actions=(
            "Continue monitoring and investigate persistent deterioration.",
        ),
        monitoring=(
            "Maintain routine BOD5 and process monitoring.",
        ),
        evidence_template=(
            "Predicted effluent BOD5 is above the normal operating range."
        ),
    ),
    DecisionRule(
        name="low_do",
        priority=3,
        contributor="Low dissolved oxygen",
        checks=(
            "Verify dissolved oxygen sensor calibration and measurement quality.",
            "Check aeration/blower operation.",
            "Review air supply and mixing conditions.",
            "Review changes in influent loading and flow.",
        ),
        actions=(
            "Investigate aeration performance before making major process changes.",
        ),
        monitoring=(
            "Increase dissolved oxygen monitoring frequency.",
        ),
        evidence_template=(
            "Measured dissolved oxygen is below the initial low-DO "
            "investigation threshold."
        ),
    ),
    DecisionRule(
        name="high_flow",
        priority=3,
        contributor="Elevated hydraulic loading",
        checks=(
            "Review current flow against the plant's normal/design flow.",
            "Check for unusual peak-flow events.",
            "Review hydraulic retention time.",
        ),
        actions=(
            "Investigate whether hydraulic loading is affecting treatment performance.",
        ),
        monitoring=(
            "Increase flow monitoring during peak periods.",
        ),
        evidence_template=(
            "Reported flow indicates elevated hydraulic loading relative "
            "to the configured baseline."
        ),
    ),
    DecisionRule(
        name="low_hrt",
        priority=3,
        contributor="Low hydraulic retention time",
        checks=(
            "Review hydraulic loading and tank operating conditions.",
            "Check whether treatment units are hydraulically overloaded.",
        ),
        actions=(
            "Investigate hydraulic performance and retention time.",
        ),
        monitoring=(
            "Monitor flow and hydraulic retention time during peak loading.",
        ),
        evidence_template=(
            "Reported hydraulic retention time is below the investigation threshold."
        ),
    ),
    DecisionRule(
        name="high_anomaly",
        priority=3,
        contributor="High process anomaly score",
        checks=(
            "Review the process measurements that produced the anomaly.",
            "Check sensors and instrumentation for abnormal readings.",
            "Compare current process conditions with recent historical values.",
        ),
        actions=(
            "Prioritize process and instrumentation investigation.",
        ),
        monitoring=(
            "Increase monitoring until the abnormal condition is explained.",
        ),
        evidence_template=(
            "The process anomaly percentile indicates an unusually abnormal "
            "operating condition."
        ),
    ),
    DecisionRule(
        name="critical_anomaly",
        priority=4,
        contributor="Critical process anomaly score",
        checks=(
            "Immediately review the process measurements contributing to the anomaly.",
            "Verify critical sensors and instrumentation.",
            "Review recent operational events and process changes.",
        ),
        actions=(
            "Escalate the condition for immediate engineering/operator review.",
        ),
        monitoring=(
            "Use increased monitoring until the abnormal condition is resolved "
            "or explained.",
        ),
        evidence_template=(
            "The process anomaly percentile is within the critical investigation range."
        ),
    ),
)


# Starting configurable thresholds.
BOD5_MODERATE_THRESHOLD = 30.0
BOD5_HIGH_THRESHOLD = 50.0
BOD5_VERY_HIGH_THRESHOLD = 80.0

LOW_DO_THRESHOLD = 1.0
LOW_HRT_THRESHOLD = 2.0

HIGH_ANOMALY_PERCENTILE = 90.0
CRITICAL_ANOMALY_PERCENTILE = 97.0

# Flow is intentionally treated as a configurable engineering trigger.
# Without a plant-specific design/baseline flow, the engine cannot
# honestly claim that a particular flow is hydraulically overloaded.
FLOW_INVESTIGATION_THRESHOLD = 1000.0
