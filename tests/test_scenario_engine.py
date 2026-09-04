from __future__ import annotations

import pytest

from decision.models import (
    DecisionPriority,
    DecisionRecommendation,
)
from risk.scoring import (
    RiskAssessment,
    RiskLevel,
)
from scenario.engine import execute_scenario
from scenario.models import (
    ScenarioInput,
    ScenarioResult,
    ScenarioStatus,
)
from scenario.validation import validate_scenario

from tests.test_scenario_models import build_scenario_input


def build_fake_risk_assessment(
    predicted_bod5: float,
    anomaly_percentile: float,
) -> RiskAssessment:
    """
    Build a deterministic risk assessment for isolated
    scenario-engine tests.
    """

    return RiskAssessment(
        risk_level=RiskLevel.LOW,
        risk_score=20.0,
        prediction_score=20.0,
        anomaly_score=10.0,
        confidence_score=100.0,
        predicted_bod5=predicted_bod5,
        anomaly_percentile=anomaly_percentile,
        risk_reason=(
            "Low predicted effluent BOD5 with normal "
            "process behaviour."
        ),
        recommended_action="Continue monitoring.",
    )


def build_fake_decision() -> DecisionRecommendation:
    """
    Build a deterministic decision recommendation.
    """

    return DecisionRecommendation(
        priority=DecisionPriority.LOW,
        summary="Continue normal monitoring.",
        possible_contributors=[
            "No major process contributor identified."
        ],
        checks_to_perform=[
            "Continue routine process checks."
        ],
        recommended_actions=[
            "Continue monitoring."
        ],
        monitoring_recommendations=[
            "Maintain routine monitoring."
        ],
        evidence=[
            "Predicted effluent BOD5 is within the expected range."
        ],
        limitations=[
            "Scenario result depends on model assumptions."
        ],
    )


def test_valid_scenario_executes_successfully():
    scenario = build_scenario_input()

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert isinstance(result, ScenarioResult)
    assert result.scenario_name == scenario.name
    assert result.predicted_effluent_bod5 == 15.0
    assert result.anomaly_score == 0.10
    assert result.anomaly_percentile == 20.0
    assert result.overall_risk_level == "LOW"
    assert result.overall_risk_score == 20.0
    assert result.decision_priority == "LOW"
    assert result.decision_summary == "Continue normal monitoring."
    assert result.status == ScenarioStatus.ANALYZED


def test_scenario_result_preserves_recommended_actions():
    scenario = build_scenario_input()

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert result.recommended_actions == [
        "Continue monitoring."
    ]


def test_scenario_result_preserves_monitoring_recommendations():
    scenario = build_scenario_input()

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert result.monitoring_recommendations == [
        "Maintain routine monitoring."
    ]


def test_scenario_result_preserves_evidence():
    scenario = build_scenario_input()

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert result.evidence == [
        "Predicted effluent BOD5 is within the expected range."
    ]


def test_scenario_result_preserves_limitations():
    scenario = build_scenario_input()

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert result.limitations == [
        "Scenario result depends on model assumptions."
    ]


def test_invalid_scenario_is_rejected_before_execution():
    scenario = build_scenario_input(
        wastewater={
            "influent_bod5": -1.0,
            "influent_cod": 100.0,
            "influent_tss": 50.0,
            "flow_m3_day": 100.0,
            "dissolved_oxygen": 2.0,
            "temperature": 25.0,
            "hrt_hours": 6.0,
        }
    )

    validation = validate_scenario(scenario)

    assert validation.valid is False
    assert validation.error_count >= 1


def test_execution_rejects_invalid_scenario():
    scenario = build_scenario_input(
        wastewater={
            "influent_bod5": -1.0,
            "influent_cod": 100.0,
            "influent_tss": 50.0,
            "flow_m3_day": 100.0,
            "dissolved_oxygen": 2.0,
            "temperature": 25.0,
            "hrt_hours": 6.0,
        }
    )

    with pytest.raises(
        ValueError,
        match="Scenario validation failed",
    ):
        execute_scenario(
            scenario,
            bod5_predictor=lambda scenario: 15.0,
            process_predictor=lambda scenario: {
                "anomaly_score": 0.10,
                "anomaly_percentile": 20.0,
            },
            risk_assessor=lambda scenario, prediction, process: (
                build_fake_risk_assessment(
                    predicted_bod5=prediction,
                    anomaly_percentile=process[
                        "anomaly_percentile"
                    ],
                )
            ),
            decision_generator=lambda scenario, assessment: (
                build_fake_decision()
            ),
        )


def test_missing_process_feature_is_rejected():
    scenario = build_scenario_input()

    scenario.process.pop("PH-P")

    validation = validate_scenario(scenario)

    assert validation.valid is False
    assert any(
        error.field == "process.PH-P"
        for error in validation.errors
    )


def test_bod5_prediction_is_passed_into_result():
    scenario = build_scenario_input()

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 42.5,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert result.predicted_effluent_bod5 == 42.5


def test_anomaly_result_is_passed_into_result():
    scenario = build_scenario_input()

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.73,
            "anomaly_percentile": 91.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert result.anomaly_score == 0.73
    assert result.anomaly_percentile == 91.0


def test_risk_assessment_is_passed_into_result():
    scenario = build_scenario_input()

    def custom_risk_assessor(
        scenario,
        prediction,
        process,
    ):
        return RiskAssessment(
            risk_level=RiskLevel.HIGH,
            risk_score=82.0,
            prediction_score=75.0,
            anomaly_score=70.0,
            confidence_score=70.0,
            predicted_bod5=prediction,
            anomaly_percentile=process["anomaly_percentile"],
            risk_reason="Elevated scenario risk.",
            recommended_action=(
                "Investigate treatment performance."
            ),
        )

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 55.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.80,
            "anomaly_percentile": 95.0,
        },
        risk_assessor=custom_risk_assessor,
        decision_generator=lambda scenario, assessment: (
            DecisionRecommendation(
                priority=DecisionPriority.HIGH,
                summary=(
                    "Investigate treatment performance."
                ),
            )
        ),
    )

    assert result.overall_risk_level == "HIGH"
    assert result.overall_risk_score == 82.0
    assert result.predicted_effluent_bod5 == 55.0


def test_decision_engine_output_is_passed_into_result():
    scenario = build_scenario_input()

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            DecisionRecommendation(
                priority=DecisionPriority.CRITICAL,
                summary=(
                    "Immediate investigation required."
                ),
                recommended_actions=[
                    "Investigate treatment failure."
                ],
                monitoring_recommendations=[
                    "Increase monitoring frequency."
                ],
                evidence=[
                    "Scenario risk is critical."
                ],
                limitations=[
                    "Model uncertainty remains."
                ],
            )
        ),
    )

    assert result.decision_priority == "CRITICAL"
    assert result.decision_summary == (
        "Immediate investigation required."
    )
    assert result.recommended_actions == [
        "Investigate treatment failure."
    ]
    assert result.monitoring_recommendations == [
        "Increase monitoring frequency."
    ]
    assert result.evidence == [
        "Scenario risk is critical."
    ]
    assert result.limitations == [
        "Model uncertainty remains."
    ]


def test_scenario_metadata_is_preserved():
    scenario = build_scenario_input()

    scenario.metadata["source"] = "V2.8.3 test"
    scenario.metadata["scenario_type"] = "baseline"

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert result.metadata["source"] == "V2.8.3 test"
    assert result.metadata["scenario_type"] == "baseline"


def test_bod5_predictor_is_called_once():
    scenario = build_scenario_input()

    calls = {"count": 0}

    def predictor(scenario):
        calls["count"] += 1
        return 15.0

    execute_scenario(
        scenario,
        bod5_predictor=predictor,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert calls["count"] == 1


def test_process_predictor_is_called_once():
    scenario = build_scenario_input()

    calls = {"count": 0}

    def predictor(scenario):
        calls["count"] += 1

        return {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        }

    execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=predictor,
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert calls["count"] == 1


def test_risk_assessor_is_called_once():
    scenario = build_scenario_input()

    calls = {"count": 0}

    def assessor(scenario, prediction, process):
        calls["count"] += 1

        return build_fake_risk_assessment(
            predicted_bod5=prediction,
            anomaly_percentile=process["anomaly_percentile"],
        )

    execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=assessor,
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert calls["count"] == 1


def test_decision_generator_is_called_once():
    scenario = build_scenario_input()

    calls = {"count": 0}

    def generator(scenario, assessment):
        calls["count"] += 1
        return build_fake_decision()

    execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=generator,
    )

    assert calls["count"] == 1


def test_bod5_prediction_failure_is_not_silently_hidden():
    scenario = build_scenario_input()

    def failing_predictor(scenario):
        raise RuntimeError("BOD5 prediction failed")

    with pytest.raises(
        RuntimeError,
        match="BOD5 prediction failed",
    ):
        execute_scenario(
            scenario,
            bod5_predictor=failing_predictor,
            process_predictor=lambda scenario: {
                "anomaly_score": 0.10,
                "anomaly_percentile": 20.0,
            },
            risk_assessor=lambda scenario, prediction, process: (
                build_fake_risk_assessment(
                    predicted_bod5=prediction,
                    anomaly_percentile=process[
                        "anomaly_percentile"
                    ],
                )
            ),
            decision_generator=lambda scenario, assessment: (
                build_fake_decision()
            ),
        )


def test_process_prediction_failure_is_not_silently_hidden():
    scenario = build_scenario_input()

    def failing_predictor(scenario):
        raise RuntimeError(
            "Process anomaly prediction failed"
        )

    with pytest.raises(
        RuntimeError,
        match="Process anomaly prediction failed",
    ):
        execute_scenario(
            scenario,
            bod5_predictor=lambda scenario: 15.0,
            process_predictor=failing_predictor,
            risk_assessor=lambda scenario, prediction, process: (
                build_fake_risk_assessment(
                    predicted_bod5=prediction,
                    anomaly_percentile=process[
                        "anomaly_percentile"
                    ],
                )
            ),
            decision_generator=lambda scenario, assessment: (
                build_fake_decision()
            ),
        )


def test_decision_failure_is_not_silently_hidden():
    scenario = build_scenario_input()

    def failing_decision_generator(
        scenario,
        assessment,
    ):
        raise RuntimeError("Decision generation failed")

    with pytest.raises(
        RuntimeError,
        match="Decision generation failed",
    ):
        execute_scenario(
            scenario,
            bod5_predictor=lambda scenario: 15.0,
            process_predictor=lambda scenario: {
                "anomaly_score": 0.10,
                "anomaly_percentile": 20.0,
            },
            risk_assessor=lambda scenario, prediction, process: (
                build_fake_risk_assessment(
                    predicted_bod5=prediction,
                    anomaly_percentile=process[
                        "anomaly_percentile"
                    ],
                )
            ),
            decision_generator=failing_decision_generator,
        )


def test_same_scenario_produces_deterministic_result():
    scenario = build_scenario_input()

    def run():
        return execute_scenario(
            scenario,
            bod5_predictor=lambda scenario: 15.0,
            process_predictor=lambda scenario: {
                "anomaly_score": 0.10,
                "anomaly_percentile": 20.0,
            },
            risk_assessor=lambda scenario, prediction, process: (
                build_fake_risk_assessment(
                    predicted_bod5=prediction,
                    anomaly_percentile=process[
                        "anomaly_percentile"
                    ],
                )
            ),
            decision_generator=lambda scenario, assessment: (
                build_fake_decision()
            ),
        )

    first = run()
    second = run()

    assert first == second


def test_different_scenarios_can_be_executed_independently():
    first_scenario = build_scenario_input(
        name="Baseline",
    )

    second_scenario = build_scenario_input(
        name="Improved aeration",
    )

    first_result = execute_scenario(
        first_scenario,
        bod5_predictor=lambda scenario: 20.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.20,
            "anomaly_percentile": 30.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    second_result = execute_scenario(
        second_scenario,
        bod5_predictor=lambda scenario: 10.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.05,
            "anomaly_percentile": 10.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert first_result.scenario_name == "Baseline"
    assert second_result.scenario_name == "Improved aeration"

    assert first_result.predicted_effluent_bod5 == 20.0
    assert second_result.predicted_effluent_bod5 == 10.0


def test_scenario_execution_does_not_require_fastapi():
    """
    The scenario engine is a domain-layer component.

    This test executes the engine directly without HTTP,
    FastAPI, or the API application.
    """

    scenario = build_scenario_input()

    result = execute_scenario(
        scenario,
        bod5_predictor=lambda scenario: 15.0,
        process_predictor=lambda scenario: {
            "anomaly_score": 0.10,
            "anomaly_percentile": 20.0,
        },
        risk_assessor=lambda scenario, prediction, process: (
            build_fake_risk_assessment(
                predicted_bod5=prediction,
                anomaly_percentile=process["anomaly_percentile"],
            )
        ),
        decision_generator=lambda scenario, assessment: (
            build_fake_decision()
        ),
    )

    assert isinstance(result, ScenarioResult)


def test_invalid_process_result_is_rejected():
    scenario = build_scenario_input()

    with pytest.raises(
        ValueError,
        match="invalid result",
    ):
        execute_scenario(
            scenario,
            bod5_predictor=lambda scenario: 15.0,
            process_predictor=lambda scenario: {},
            risk_assessor=lambda scenario, prediction, process: (
                build_fake_risk_assessment(
                    predicted_bod5=prediction,
                    anomaly_percentile=20.0,
                )
            ),
            decision_generator=lambda scenario, assessment: (
                build_fake_decision()
            ),
        )


def test_invalid_risk_assessor_output_is_rejected():
    scenario = build_scenario_input()

    with pytest.raises(
        TypeError,
        match="risk_assessor must return a RiskAssessment",
    ):
        execute_scenario(
            scenario,
            bod5_predictor=lambda scenario: 15.0,
            process_predictor=lambda scenario: {
                "anomaly_score": 0.10,
                "anomaly_percentile": 20.0,
            },
            risk_assessor=lambda scenario, prediction, process: (
                "invalid"
            ),
            decision_generator=lambda scenario, assessment: (
                build_fake_decision()
            ),
        )


def test_invalid_decision_generator_output_is_rejected():
    scenario = build_scenario_input()

    with pytest.raises(
        TypeError,
        match="decision_generator must return",
    ):
        execute_scenario(
            scenario,
            bod5_predictor=lambda scenario: 15.0,
            process_predictor=lambda scenario: {
                "anomaly_score": 0.10,
                "anomaly_percentile": 20.0,
            },
            risk_assessor=lambda scenario, prediction, process: (
                build_fake_risk_assessment(
                    predicted_bod5=prediction,
                    anomaly_percentile=20.0,
                )
            ),
            decision_generator=lambda scenario, assessment: (
                "invalid"
            ),
        )
