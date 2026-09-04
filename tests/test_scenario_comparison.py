from __future__ import annotations

import pytest

from scenario.comparison import compare_scenarios
from scenario.models import (
    ScenarioResult,
    ScenarioStatus,
)


def build_result(
    *,
    name: str = "Scenario A",
    bod5: float = 15.0,
    anomaly_score: float = 0.10,
    anomaly_percentile: float = 20.0,
    risk_level: str = "NORMAL",
    risk_score: float = 10.0,
    priority: str = "NORMAL",
    summary: str = "Normal treatment condition.",
) -> ScenarioResult:
    """
    Build a valid ScenarioResult for comparison tests.
    Individual arguments can be overridden by tests so that
    each comparison rule can be tested independently.
    """
    return ScenarioResult(
        scenario_name=name,
        predicted_effluent_bod5=bod5,
        anomaly_score=anomaly_score,
        anomaly_percentile=anomaly_percentile,
        overall_risk_level=risk_level,
        overall_risk_score=risk_score,
        decision_priority=priority,
        decision_summary=summary,
        recommended_actions=[
            "Continue routine monitoring."
        ],
        monitoring_recommendations=[
            "Continue routine process monitoring."
        ],
        evidence=[
            "Prediction and process monitoring results."
        ],
        limitations=[
            "Model results require operational validation."
        ],
        status=ScenarioStatus.ANALYZED,
    )


def test_compare_scenarios_requires_at_least_one_scenario():
    with pytest.raises(
        ValueError,
        match="at least one scenario result is required",
    ):
        compare_scenarios([])


def test_compare_scenarios_returns_comparison():
    scenario = build_result()
    comparison = compare_scenarios([scenario])

    assert comparison.scenario_count == 1
    assert comparison.preferred_scenario == "Scenario A"


def test_single_scenario_is_preferred():
    scenario = build_result(name="Baseline")
    comparison = compare_scenarios([scenario])

    assert comparison.preferred_scenario == "Baseline"


def test_lower_risk_level_is_preferred_over_higher_risk():
    normal = build_result(
        name="Normal",
        bod5=18.0,
        risk_level="NORMAL",
        risk_score=20.0,
    )
    high = build_result(
        name="High",
        bod5=10.0,
        risk_level="HIGH",
        risk_score=5.0,
    )

    comparison = compare_scenarios([high, normal])

    assert comparison.preferred_scenario == "Normal"


def test_lower_risk_score_is_preferred_when_risk_level_matches():
    first = build_result(
        name="Scenario A",
        risk_level="ELEVATED",
        risk_score=40.0,
        bod5=15.0,
    )
    second = build_result(
        name="Scenario B",
        risk_level="ELEVATED",
        risk_score=25.0,
        bod5=20.0,
    )

    comparison = compare_scenarios([first, second])

    assert comparison.preferred_scenario == "Scenario B"


def test_lower_bod5_is_used_after_risk_level_and_score():
    first = build_result(
        name="Scenario A",
        risk_level="LOW",
        risk_score=20.0,
        bod5=20.0,
    )
    second = build_result(
        name="Scenario B",
        risk_level="LOW",
        risk_score=20.0,
        bod5=12.0,
    )

    comparison = compare_scenarios([first, second])

    assert comparison.preferred_scenario == "Scenario B"


def test_decision_priority_is_final_tiebreaker():
    first = build_result(
        name="Scenario A",
        risk_level="LOW",
        risk_score=20.0,
        bod5=15.0,
        priority="HIGH",
    )
    second = build_result(
        name="Scenario B",
        risk_level="LOW",
        risk_score=20.0,
        bod5=15.0,
        priority="LOW",
    )

    comparison = compare_scenarios([first, second])

    assert comparison.preferred_scenario == "Scenario B"


def test_comparison_does_not_depend_on_input_order():
    first = build_result(
        name="Scenario A",
        risk_level="LOW",
        risk_score=30.0,
        bod5=20.0,
    )
    second = build_result(
        name="Scenario B",
        risk_level="NORMAL",
        risk_score=50.0,
        bod5=25.0,
    )

    result_one = compare_scenarios([first, second])

    result_two = compare_scenarios([second, first])

    assert (
        result_one.preferred_scenario
        == result_two.preferred_scenario
    )


def test_comparison_preserves_original_scenario_results():
    first = build_result(name="Scenario A")
    second = build_result(
        name="Scenario B",
        bod5=12.0,
    )

    comparison = compare_scenarios([first, second])

    assert comparison.scenarios == [
        first,
        second,
    ]


def test_selection_reason_is_provided():
    first = build_result(name="Baseline")
    second = build_result(
        name="Improved",
        bod5=10.0,
        risk_score=5.0,
    )

    comparison = compare_scenarios([first, second])

    assert comparison.selection_reason is not None
    assert "Improved" in comparison.selection_reason


def test_tradeoffs_are_generated_for_alternatives():
    first = build_result(
        name="Baseline",
        bod5=20.0,
        risk_score=30.0,
    )
    second = build_result(
        name="Improved",
        bod5=12.0,
        risk_score=10.0,
    )

    comparison = compare_scenarios([first, second])

    assert comparison.preferred_scenario == "Improved"
    assert comparison.trade_offs

    assert any(
        "Baseline" in trade_off
        for trade_off in comparison.trade_offs
    )


def test_comparison_metadata_describes_method():
    scenario = build_result()
    comparison = compare_scenarios([scenario])

    assert comparison.metadata[
        "comparison_method"
    ] == "risk_first"

    assert comparison.metadata[
        "ranking_order"
    ] == [
        "overall_risk_level",
        "overall_risk_score",
        "predicted_effluent_bod5",
        "decision_priority",
    ]


def test_duplicate_scenario_names_are_rejected():
    first = build_result(name="Baseline")
    second = build_result(
        name="Baseline",
        bod5=12.0,
    )

    with pytest.raises(
        ValueError,
        match="duplicate scenario name",
    ):
        compare_scenarios([first, second])


def test_invalid_scenario_result_type_is_rejected():
    with pytest.raises(
        TypeError,
        match="all scenarios must be ScenarioResult instances",
    ):
        compare_scenarios([object()])  # type: ignore[list-item]


def test_negative_bod5_is_rejected():
    scenario = build_result(bod5=-1.0)
    with pytest.raises(
        ValueError,
        match="predicted BOD5 must not be negative",
    ):
        compare_scenarios([scenario])


def test_negative_anomaly_score_is_rejected():
    scenario = build_result(anomaly_score=-0.1)
    with pytest.raises(
        ValueError,
        match="anomaly score must not be negative",
    ):
        compare_scenarios([scenario])


def test_invalid_anomaly_percentile_is_rejected():
    scenario = build_result(anomaly_percentile=101.0)
    with pytest.raises(
        ValueError,
        match="anomaly percentile must be between 0 and 100",
    ):
        compare_scenarios([scenario])


def test_negative_risk_score_is_rejected():
    scenario = build_result(risk_score=-1.0)
    with pytest.raises(
        ValueError,
        match="overall risk score must not be negative",
    ):
        compare_scenarios([scenario])


def test_empty_risk_level_is_rejected():
    scenario = build_result(risk_level="")
    with pytest.raises(
        ValueError,
        match="overall risk level must not be empty",
    ):
        compare_scenarios([scenario])


def test_unsupported_risk_level_is_rejected():
    scenario = build_result(risk_level="UNKNOWN")
    with pytest.raises(
        ValueError,
        match="unsupported overall risk level",
    ):
        compare_scenarios([scenario])


def test_empty_decision_priority_is_rejected():
    scenario = build_result(priority="")
    with pytest.raises(
        ValueError,
        match="decision priority must not be empty",
    ):
        compare_scenarios([scenario])


def test_unsupported_decision_priority_is_rejected():
    scenario = build_result(priority="UNKNOWN")
    with pytest.raises(
        ValueError,
        match="unsupported decision priority",
    ):
        compare_scenarios([scenario])


def test_comparison_is_deterministic():
    scenarios = [
        build_result(
            name="Scenario A",
            risk_level="ELEVATED",
            risk_score=40.0,
            bod5=18.0,
        ),
        build_result(
            name="Scenario B",
            risk_level="LOW",
            risk_score=20.0,
            bod5=15.0,
        ),
        build_result(
            name="Scenario C",
            risk_level="NORMAL",
            risk_score=30.0,
            bod5=12.0,
        ),
    ]
    first = compare_scenarios(scenarios)

    second = compare_scenarios(scenarios)

    assert first == second
