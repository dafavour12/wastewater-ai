from __future__ import annotations

from scenario.models import (
    ScenarioComparison,
    ScenarioResult,
    ScenarioStatus,
)


def _validate_scenario_results(
    scenarios: list[ScenarioResult],
) -> None:
    """
    Validate the scenario results supplied for comparison.
    """

    if not scenarios:
        raise ValueError(
            "at least one scenario result is required"
        )

    seen_names: set[str] = set()

    for scenario in scenarios:
        if not isinstance(
            scenario,
            ScenarioResult,
        ):
            raise TypeError(
                "all scenarios must be ScenarioResult instances"
            )

        name = scenario.scenario_name.strip()

        if not name:
            raise ValueError(
                "scenario name must not be empty"
            )

        if name in seen_names:
            raise ValueError(
                f"duplicate scenario name: {name}"
            )

        seen_names.add(name)

        if scenario.predicted_effluent_bod5 < 0:
            raise ValueError(
                f"predicted BOD5 must not be negative "
                f"for scenario '{name}'"
            )

        if scenario.anomaly_score < 0:
            raise ValueError(
                f"anomaly score must not be negative "
                f"for scenario '{name}'"
            )

        if not 0.0 <= scenario.anomaly_percentile <= 100.0:
            raise ValueError(
                f"anomaly percentile must be between 0 and 100 "
                f"for scenario '{name}'"
            )

        if scenario.overall_risk_score < 0:
            raise ValueError(
                f"overall risk score must not be negative "
                f"for scenario '{name}'"
            )

        if not scenario.overall_risk_level.strip():
            raise ValueError(
                f"overall risk level must not be empty "
                f"for scenario '{name}'"
            )

        if not scenario.decision_priority.strip():
            raise ValueError(
                f"decision priority must not be empty "
                f"for scenario '{name}'"
            )


def _risk_level_rank(
    risk_level: str,
) -> int:
    """
    Convert a risk level into its deterministic comparison rank.

    Lower rank means lower risk and therefore a better scenario.
    """

    ranks = {
        "NORMAL": 0,
        "LOW": 1,
        "ELEVATED": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    normalized = risk_level.strip().upper()

    if normalized not in ranks:
        raise ValueError(
            f"unsupported overall risk level: {risk_level}"
        )

    return ranks[normalized]


def _decision_priority_rank(
    priority: str,
) -> int:
    """
    Convert a decision priority into its deterministic rank.

    Lower rank means lower intervention priority.
    """

    ranks = {
        "NORMAL": 0,
        "LOW": 1,
        "MODERATE": 2,
        "HIGH": 3,
        "CRITICAL": 4,
    }

    normalized = priority.strip().upper()

    if normalized not in ranks:
        raise ValueError(
            f"unsupported decision priority: {priority}"
        )

    return ranks[normalized]


def _comparison_key(
    scenario: ScenarioResult,
) -> tuple[int, float, float, int]:
    """
    Build the deterministic ranking key for one scenario.

    Comparison priority:

    1. overall risk level
    2. overall risk score
    3. predicted effluent BOD5
    4. decision priority
    """

    return (
        _risk_level_rank(
            scenario.overall_risk_level
        ),
        scenario.overall_risk_score,
        scenario.predicted_effluent_bod5,
        _decision_priority_rank(
            scenario.decision_priority
        ),
    )


def _build_selection_reason(
    preferred: ScenarioResult,
    scenarios: list[ScenarioResult],
) -> str:
    """
    Explain why the preferred scenario was selected.
    """

    if len(scenarios) == 1:
        return (
            f"'{preferred.scenario_name}' is the only scenario "
            "provided for comparison."
        )

    return (
        f"'{preferred.scenario_name}' was preferred because it "
        "has the best overall comparison ranking, prioritizing "
        "lower risk level, lower risk score, lower predicted "
        "effluent BOD5, and lower decision priority."
    )


def _build_trade_offs(
    preferred: ScenarioResult,
    scenarios: list[ScenarioResult],
) -> list[str]:
    """
    Identify important differences between the preferred
    scenario and the alternatives.

    These are observations, not optimization conclusions.
    """

    trade_offs: list[str] = []

    for scenario in scenarios:
        if scenario.scenario_name == preferred.scenario_name:
            continue

        if (
            scenario.overall_risk_level.strip().upper()
            != preferred.overall_risk_level.strip().upper()
        ):
            trade_offs.append(
                f"'{scenario.scenario_name}' has a different "
                "overall risk level from the preferred scenario."
            )

        if (
            scenario.predicted_effluent_bod5
            != preferred.predicted_effluent_bod5
        ):
            trade_offs.append(
                f"'{scenario.scenario_name}' has a different "
                "predicted effluent BOD5."
            )

        if (
            scenario.overall_risk_score
            != preferred.overall_risk_score
        ):
            trade_offs.append(
                f"'{scenario.scenario_name}' has a different "
                "overall risk score."
            )

    return trade_offs


def compare_scenarios(
    scenarios: list[ScenarioResult],
) -> ScenarioComparison:
    """
    Compare multiple evaluated wastewater treatment scenarios.

    The comparison is deterministic and does not rerun any
    prediction, anomaly, risk, or decision models.

    Lower risk is preferred over lower BOD5. This prevents a
    scenario with a slightly better predicted BOD5 from being
    selected when its overall risk is substantially worse.
    """

    _validate_scenario_results(scenarios)

    ranked_scenarios = sorted(
        scenarios,
        key=_comparison_key,
    )

    preferred = ranked_scenarios[0]

    return ScenarioComparison(
        scenarios=list(scenarios),
        preferred_scenario=preferred.scenario_name,
        selection_reason=_build_selection_reason(
            preferred,
            scenarios,
        ),
        trade_offs=_build_trade_offs(
            preferred,
            scenarios,
        ),
        metadata={
            "comparison_method": "risk_first",
            "ranking_order": [
                "overall_risk_level",
                "overall_risk_score",
                "predicted_effluent_bod5",
                "decision_priority",
            ],
        },
    )