from __future__ import annotations

from collections.abc import Callable
from typing import Any

from decision.models import (
    DecisionRecommendation,
)
from risk.scoring import RiskAssessment
from scenario.models import (
    ScenarioInput,
    ScenarioResult,
    ScenarioStatus,
)
from scenario.validation import validate_scenario


# ---------------------------------------------------------------------------
# V2.8.3 execution function types
# ---------------------------------------------------------------------------

BOD5Predictor = Callable[
    [ScenarioInput],
    float,
]

ProcessPredictor = Callable[
    [ScenarioInput],
    Any,
]

RiskAssessor = Callable[
    [ScenarioInput, float, Any],
    RiskAssessment,
]

DecisionGenerator = Callable[
    [ScenarioInput, RiskAssessment],
    DecisionRecommendation,
]


# ---------------------------------------------------------------------------
# Scenario execution
# ---------------------------------------------------------------------------

def execute_scenario(
    scenario: ScenarioInput,
    *,
    bod5_predictor: BOD5Predictor,
    process_predictor: ProcessPredictor,
    risk_assessor: RiskAssessor,
    decision_generator: DecisionGenerator,
) -> ScenarioResult:
    """
    Execute one wastewater treatment scenario.

    The scenario engine is an orchestration layer.

    It does not reimplement:
    - BOD5 machine-learning prediction
    - process anomaly detection
    - risk scoring
    - engineering decision rules

    Instead, it coordinates those existing capabilities and converts
    their outputs into one ScenarioResult.

    FastAPI is intentionally not required here.
    """

    # -----------------------------------------------------------------------
    # 1. Validate scenario
    # -----------------------------------------------------------------------

    validation = validate_scenario(scenario)

    if not validation.valid:
        messages = [
            f"{error.field}: {error.message}"
            for error in validation.errors
        ]

        raise ValueError(
            "Scenario validation failed: "
            + "; ".join(messages)
        )

    # -----------------------------------------------------------------------
    # 2. Run BOD5 prediction
    # -----------------------------------------------------------------------

    predicted_bod5 = bod5_predictor(scenario)

    # -----------------------------------------------------------------------
    # 3. Run process anomaly detection
    # -----------------------------------------------------------------------

    process_result = process_predictor(scenario)

    try:
        anomaly_score = float(
            process_result["anomaly_score"]
        )
        anomaly_percentile = float(
            process_result["anomaly_percentile"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            "Process predictor returned an invalid result."
        ) from exc

    # -----------------------------------------------------------------------
    # 4. Run combined risk assessment
    # -----------------------------------------------------------------------

    assessment = risk_assessor(
        scenario,
        predicted_bod5,
        process_result,
    )

    if not isinstance(assessment, RiskAssessment):
        raise TypeError(
            "risk_assessor must return a RiskAssessment."
        )

    # -----------------------------------------------------------------------
    # 5. Generate engineering decision
    # -----------------------------------------------------------------------

    decision = decision_generator(
        scenario,
        assessment,
    )

    if not isinstance(
        decision,
        DecisionRecommendation,
    ):
        raise TypeError(
            "decision_generator must return a "
            "DecisionRecommendation."
        )

    # -----------------------------------------------------------------------
    # 6. Build scenario result
    # -----------------------------------------------------------------------

    return ScenarioResult(
        scenario_name=scenario.name,
        predicted_effluent_bod5=round(
            predicted_bod5,
            2,
        ),
        anomaly_score=anomaly_score,
        anomaly_percentile=anomaly_percentile,
        overall_risk_level=(
            assessment.risk_level.name
        ),
        overall_risk_score=round(
            assessment.risk_score,
            2,
        ),
        decision_priority=(
            decision.priority.name
        ),
        decision_summary=decision.summary,
        recommended_actions=list(
            decision.recommended_actions
        ),
        monitoring_recommendations=list(
            decision.monitoring_recommendations
        ),
        evidence=list(
            decision.evidence
        ),
        limitations=list(
            decision.limitations
        ),
        status=ScenarioStatus.ANALYZED,
        metadata=dict(
            scenario.metadata
        ),
    )
