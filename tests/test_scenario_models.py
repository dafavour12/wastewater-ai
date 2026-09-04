from __future__ import annotations

import pytest

from scenario.models import (
    OptimizationConstraint,
    ScenarioComparison,
    ScenarioInput,
    ScenarioResult,
    ScenarioStatus,
)


def build_scenario_input(
    *,
    name: str = "Baseline Scenario",
    description: str = "Baseline wastewater treatment scenario.",
    wastewater: dict[str, float] | None = None,
    process: dict[str, float] | None = None,
    model_confidence: str = "research",
    metadata: dict | None = None,
) -> ScenarioInput:
    """
    Build a valid ScenarioInput for tests.

    Optional arguments allow individual tests to override
    specific scenario values without duplicating the entire
    scenario definition.
    """

    default_wastewater = {
        "influent_bod5": 300.0,
        "influent_cod": 600.0,
        "influent_tss": 250.0,
        "flow_m3_day": 1000.0,
        "dissolved_oxygen": 2.1,
        "temperature": 25.0,
        "hrt_hours": 8.0,
    }

    default_process = {
        "PH-P": 7.0,
        "DBO-P": 300.0,
        "SS-P": 250.0,
        "SSV-P": 150.0,
        "SED-P": 100.0,
        "COND-P": 500.0,
        "PH-D": 7.2,
        "DBO-D": 20.0,
        "DQO-D": 50.0,
        "SS-D": 30.0,
        "SSV-D": 20.0,
        "SED-D": 10.0,
        "COND-D": 450.0,
        "RD-DBO-P": 0.10,
        "RD-SS-P": 0.10,
        "RD-DBO-D": 0.05,
        "RD-SS-D": 0.05,
        "RD-DBO-G": 0.10,
        "RD-SS-G": 0.10,
        "RD-SED-G": 0.05,
        "RD-N-NH4": 0.05,
        "RD-N-NO2": 0.02,
    }

    default_metadata = {
        "source": "test",
        "scenario_type": "baseline",
    }

    return ScenarioInput(
        name=name,
        description=description,
        wastewater=(
            default_wastewater
            if wastewater is None
            else wastewater
        ),
        process=(
            default_process
            if process is None
            else process
        ),
        model_confidence=model_confidence,
        metadata=(
            default_metadata
            if metadata is None
            else metadata
        ),
    )


def build_scenario_result(
    name: str = "Baseline Scenario",
) -> ScenarioResult:
    return ScenarioResult(
        scenario_name=name,
        predicted_effluent_bod5=15.0,
        anomaly_score=0.10,
        anomaly_percentile=20.0,
        overall_risk_level="low",
        overall_risk_score=25.0,
        decision_priority="low",
        decision_summary="Continue monitoring treatment performance.",
        recommended_actions=[
            "Continue routine monitoring.",
        ],
        monitoring_recommendations=[
            "Monitor effluent BOD5.",
        ],
        evidence=[
            "Predicted BOD5 is within the expected operating range.",
        ],
        limitations=[
            "Model output should be interpreted with process data.",
        ],
    )


def test_scenario_input_stores_required_data():
    scenario = build_scenario_input()

    assert scenario.name == "Baseline Scenario"
    assert scenario.description
    assert scenario.model_confidence == "research"

    assert scenario.wastewater["influent_bod5"] == 300.0
    assert scenario.wastewater["dissolved_oxygen"] == 2.1
    assert scenario.wastewater["hrt_hours"] == 8.0

    assert scenario.process["PH-P"] == 7.0
    assert len(scenario.process) == 22


def test_scenario_input_supports_metadata():
    scenario = build_scenario_input()

    assert scenario.metadata["source"] == "test"
    assert scenario.metadata["scenario_type"] == "baseline"


def test_scenario_input_is_immutable():
    scenario = build_scenario_input()

    with pytest.raises(AttributeError):
        scenario.name = "Changed"


def test_optimization_constraint_supports_minimum_only():
    constraint = OptimizationConstraint(
        parameter="dissolved_oxygen",
        minimum=2.0,
        unit="mg/L",
    )

    assert constraint.parameter == "dissolved_oxygen"
    assert constraint.minimum == 2.0
    assert constraint.maximum is None
    assert constraint.unit == "mg/L"


def test_optimization_constraint_supports_maximum_only():
    constraint = OptimizationConstraint(
        parameter="flow_m3_day",
        maximum=1500.0,
        unit="m3/day",
    )

    assert constraint.parameter == "flow_m3_day"
    assert constraint.minimum is None
    assert constraint.maximum == 1500.0


def test_optimization_constraint_supports_minimum_and_maximum():
    constraint = OptimizationConstraint(
        parameter="hrt_hours",
        minimum=6.0,
        maximum=24.0,
        unit="hours",
    )

    assert constraint.minimum == 6.0
    assert constraint.maximum == 24.0


def test_optimization_constraint_rejects_empty_parameter():
    with pytest.raises(ValueError, match="parameter"):
        OptimizationConstraint(
            parameter="   ",
            minimum=1.0,
        )


def test_optimization_constraint_requires_boundary():
    with pytest.raises(
        ValueError,
        match="minimum or maximum",
    ):
        OptimizationConstraint(
            parameter="dissolved_oxygen",
        )


def test_optimization_constraint_rejects_invalid_range():
    with pytest.raises(
        ValueError,
        match="minimum must not be greater",
    ):
        OptimizationConstraint(
            parameter="hrt_hours",
            minimum=24.0,
            maximum=6.0,
        )


def test_scenario_result_stores_analysis_output():
    result = build_scenario_result()

    assert result.scenario_name == "Baseline Scenario"
    assert result.predicted_effluent_bod5 == 15.0
    assert result.anomaly_score == 0.10
    assert result.anomaly_percentile == 20.0

    assert result.overall_risk_level == "low"
    assert result.overall_risk_score == 25.0

    assert result.decision_priority == "low"
    assert result.decision_summary

    assert isinstance(result.recommended_actions, list)
    assert isinstance(
        result.monitoring_recommendations,
        list,
    )
    assert isinstance(result.evidence, list)
    assert isinstance(result.limitations, list)


def test_scenario_result_defaults_to_analyzed_status():
    result = build_scenario_result()

    assert result.status == ScenarioStatus.ANALYZED


def test_scenario_comparison_stores_scenarios():
    first = build_scenario_result("Baseline")
    second = build_scenario_result("Improved DO")

    comparison = ScenarioComparison(
        scenarios=[first, second],
        preferred_scenario="Improved DO",
        selection_reason="Lower predicted BOD5.",
        trade_offs=[
            "Higher aeration demand may increase energy use."
        ],
    )

    assert comparison.scenario_count == 2
    assert comparison.preferred_scenario == "Improved DO"
    assert comparison.selection_reason == "Lower predicted BOD5."

    assert len(comparison.scenarios) == 2
    assert len(comparison.trade_offs) == 1


def test_empty_scenario_comparison_is_supported():
    comparison = ScenarioComparison()

    assert comparison.scenario_count == 0
    assert comparison.preferred_scenario is None
    assert comparison.selection_reason is None
    assert comparison.trade_offs == []
