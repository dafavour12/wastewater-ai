from __future__ import annotations

import math

from scenario.models import OptimizationConstraint
from scenario.validation import (
    REQUIRED_PROCESS_FIELDS,
    REQUIRED_WASTEWATER_FIELDS,
    validate_optimization_constraint,
    validate_optimization_constraints,
    validate_scenario,
)

from tests.test_scenario_models import build_scenario_input


def test_valid_scenario_passes_validation():
    scenario = build_scenario_input()

    result = validate_scenario(scenario)

    assert result.valid is True
    assert result.errors == []
    assert result.error_count == 0


def test_required_wastewater_fields_are_defined():
    expected = {
        "influent_bod5",
        "influent_cod",
        "influent_tss",
        "flow_m3_day",
        "dissolved_oxygen",
        "temperature",
        "hrt_hours",
    }

    assert REQUIRED_WASTEWATER_FIELDS == expected


def test_required_process_fields_contain_22_features():
    assert len(REQUIRED_PROCESS_FIELDS) == 22


def test_missing_wastewater_field_is_rejected():
    scenario = build_scenario_input()

    scenario.wastewater.pop("influent_bod5")

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        error.field == "wastewater.influent_bod5"
        for error in result.errors
    )


def test_missing_process_feature_is_rejected():
    scenario = build_scenario_input()

    scenario.process.pop("PH-P")

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        error.field == "process.PH-P"
        for error in result.errors
    )


def test_non_numeric_wastewater_value_is_rejected():
    scenario = build_scenario_input()

    scenario.wastewater["influent_bod5"] = "300"

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        error.field == "wastewater.influent_bod5"
        and "numeric" in error.message
        for error in result.errors
    )


def test_nan_wastewater_value_is_rejected():
    scenario = build_scenario_input()

    scenario.wastewater["influent_bod5"] = math.nan

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        error.field == "wastewater.influent_bod5"
        and "finite" in error.message
        for error in result.errors
    )


def test_infinite_process_value_is_rejected():
    scenario = build_scenario_input()

    scenario.process["PH-P"] = math.inf

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        error.field == "process.PH-P"
        and "finite" in error.message
        for error in result.errors
    )


def test_bod5_above_scenario_limit_is_rejected():
    scenario = build_scenario_input()

    scenario.wastewater["influent_bod5"] = 5001.0

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        error.field == "wastewater.influent_bod5"
        for error in result.errors
    )


def test_negative_flow_is_rejected():
    scenario = build_scenario_input()

    scenario.wastewater["flow_m3_day"] = -1.0

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        error.field == "wastewater.flow_m3_day"
        for error in result.errors
    )


def test_excessive_dissolved_oxygen_is_rejected():
    scenario = build_scenario_input()

    scenario.wastewater["dissolved_oxygen"] = 20.1

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        error.field == "wastewater.dissolved_oxygen"
        for error in result.errors
    )


def test_bod5_greater_than_cod_is_rejected():
    scenario = build_scenario_input()

    scenario.wastewater["influent_bod5"] = 600.0
    scenario.wastewater["influent_cod"] = 500.0

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        "BOD5 must not exceed" in error.message
        for error in result.errors
    )


def test_empty_scenario_name_is_rejected():
    scenario = build_scenario_input()

    scenario = scenario.__class__(
        name="   ",
        description=scenario.description,
        wastewater=scenario.wastewater,
        process=scenario.process,
        model_confidence=scenario.model_confidence,
        metadata=scenario.metadata,
    )

    result = validate_scenario(scenario)

    assert result.valid is False

    assert any(
        error.field == "name"
        for error in result.errors
    )


def test_valid_minimum_only_constraint_passes():
    constraint = OptimizationConstraint(
        parameter="dissolved_oxygen",
        minimum=2.0,
        unit="mg/L",
    )

    result = validate_optimization_constraint(constraint)

    assert result.valid is True
    assert result.errors == []


def test_valid_maximum_only_constraint_passes():
    constraint = OptimizationConstraint(
        parameter="flow_m3_day",
        maximum=1500.0,
        unit="m3/day",
    )

    result = validate_optimization_constraint(constraint)

    assert result.valid is True


def test_constraint_with_invalid_range_is_rejected():
    import pytest

    with pytest.raises(
        ValueError,
        match="minimum must not be greater than maximum",
    ):
        OptimizationConstraint(
            parameter="hrt_hours",
            minimum=24.0,
            maximum=6.0,
            unit="hours",
        )


def test_constraint_with_nan_is_rejected():
    constraint = OptimizationConstraint(
        parameter="hrt_hours",
        minimum=math.nan,
    )

    result = validate_optimization_constraint(constraint)

    assert result.valid is False

    assert any(
        "finite" in error.message
        for error in result.errors
    )


def test_duplicate_constraints_are_rejected():
    constraints = [
        OptimizationConstraint(
            parameter="dissolved_oxygen",
            minimum=2.0,
        ),
        OptimizationConstraint(
            parameter="dissolved_oxygen",
            maximum=5.0,
        ),
    ]

    result = validate_optimization_constraints(constraints)

    assert result.valid is False

    assert any(
        "duplicate parameter" in error.message
        for error in result.errors
    )


def test_multiple_valid_constraints_pass():
    constraints = [
        OptimizationConstraint(
            parameter="dissolved_oxygen",
            minimum=2.0,
            maximum=5.0,
        ),
        OptimizationConstraint(
            parameter="flow_m3_day",
            maximum=1500.0,
        ),
        OptimizationConstraint(
            parameter="hrt_hours",
            minimum=6.0,
        ),
    ]

    result = validate_optimization_constraints(constraints)

    assert result.valid is True
    assert result.errors == []
