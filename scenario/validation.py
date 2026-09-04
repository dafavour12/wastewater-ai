from __future__ import annotations

import math
from dataclasses import dataclass

from scenario.models import (
    OptimizationConstraint,
    ScenarioInput,
)


@dataclass(frozen=True)
class ScenarioValidationError:
    """
    Describes one validation problem found in a scenario.
    """

    field: str
    message: str
    value: object | None = None


@dataclass(frozen=True)
class ScenarioValidationResult:
    """
    Result returned after validating a scenario.

    A scenario is valid only when errors is empty.
    """

    valid: bool
    errors: list[ScenarioValidationError]

    @property
    def error_count(self) -> int:
        return len(self.errors)


# Core wastewater fields required by the existing V2.7 workflow.
REQUIRED_WASTEWATER_FIELDS = {
    "influent_bod5",
    "influent_cod",
    "influent_tss",
    "flow_m3_day",
    "dissolved_oxygen",
    "temperature",
    "hrt_hours",
}


# The process model requires exactly these 22 features.
REQUIRED_PROCESS_FIELDS = {
    "PH-P",
    "DBO-P",
    "SS-P",
    "SSV-P",
    "SED-P",
    "COND-P",
    "PH-D",
    "DBO-D",
    "DQO-D",
    "SS-D",
    "SSV-D",
    "SED-D",
    "COND-D",
    "RD-DBO-P",
    "RD-SS-P",
    "RD-DBO-D",
    "RD-SS-D",
    "RD-DBO-G",
    "RD-SS-G",
    "RD-SED-G",
    "RD-N-NH4",
    "RD-N-NO2",
}


# Scenario-level physical limits.
#
# These are intentionally aligned with the existing API guardrails.
SCENARIO_LIMITS = {
    "influent_bod5": (0.0, 5000.0),
    "influent_cod": (0.0, 10000.0),
    "influent_tss": (0.0, 5000.0),
    "flow_m3_day": (0.000001, 100000.0),
    "dissolved_oxygen": (0.0, 20.0),
    "temperature": (0.0, 60.0),
    "hrt_hours": (0.000001, 1000.0),
}


def _validate_finite_numeric(
    field: str,
    value: object,
    errors: list[ScenarioValidationError],
) -> None:
    """
    Validate that a value is numeric and finite.
    """

    if isinstance(value, bool) or not isinstance(
        value,
        (int, float),
    ):
        errors.append(
            ScenarioValidationError(
                field=field,
                message="value must be numeric",
                value=value,
            )
        )
        return

    if not math.isfinite(float(value)):
        errors.append(
            ScenarioValidationError(
                field=field,
                message="value must be finite",
                value=value,
            )
        )


def _validate_wastewater(
    scenario: ScenarioInput,
    errors: list[ScenarioValidationError],
) -> None:
    """
    Validate wastewater scenario inputs.
    """

    wastewater = scenario.wastewater

    missing = REQUIRED_WASTEWATER_FIELDS - wastewater.keys()

    for field in sorted(missing):
        errors.append(
            ScenarioValidationError(
                field=f"wastewater.{field}",
                message="required wastewater field is missing",
            )
        )

    for field, value in wastewater.items():
        _validate_finite_numeric(
            f"wastewater.{field}",
            value,
            errors,
        )

    for field, (minimum, maximum) in SCENARIO_LIMITS.items():
        if field not in wastewater:
            continue

        value = wastewater[field]

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            continue

        numeric_value = float(value)

        if not math.isfinite(numeric_value):
            continue

        if numeric_value < minimum:
            errors.append(
                ScenarioValidationError(
                    field=f"wastewater.{field}",
                    message=(
                        f"value must be greater than or equal to "
                        f"{minimum}"
                    ),
                    value=value,
                )
            )

        if numeric_value > maximum:
            errors.append(
                ScenarioValidationError(
                    field=f"wastewater.{field}",
                    message=(
                        f"value must be less than or equal to "
                        f"{maximum}"
                    ),
                    value=value,
                )
            )

    bod5 = wastewater.get("influent_bod5")
    cod = wastewater.get("influent_cod")

    if (
        isinstance(bod5, (int, float))
        and not isinstance(bod5, bool)
        and isinstance(cod, (int, float))
        and not isinstance(cod, bool)
        and math.isfinite(float(bod5))
        and math.isfinite(float(cod))
        and float(bod5) > float(cod)
    ):
        errors.append(
            ScenarioValidationError(
                field="wastewater.influent_bod5",
                message="influent BOD5 must not exceed influent COD",
                value=bod5,
            )
        )


def _validate_process(
    scenario: ScenarioInput,
    errors: list[ScenarioValidationError],
) -> None:
    """
    Validate the process feature set required by the anomaly model.
    """

    process = scenario.process

    missing = REQUIRED_PROCESS_FIELDS - process.keys()

    for field in sorted(missing):
        errors.append(
            ScenarioValidationError(
                field=f"process.{field}",
                message="required process feature is missing",
            )
        )

    for field, value in process.items():
        _validate_finite_numeric(
            f"process.{field}",
            value,
            errors,
        )


def _validate_metadata(
    scenario: ScenarioInput,
    errors: list[ScenarioValidationError],
) -> None:
    """
    Validate scenario identity and metadata.
    """

    if not isinstance(scenario.name, str):
        errors.append(
            ScenarioValidationError(
                field="name",
                message="scenario name must be a string",
                value=scenario.name,
            )
        )
    elif not scenario.name.strip():
        errors.append(
            ScenarioValidationError(
                field="name",
                message="scenario name must not be empty",
                value=scenario.name,
            )
        )

    if not isinstance(scenario.description, str):
        errors.append(
            ScenarioValidationError(
                field="description",
                message="scenario description must be a string",
                value=scenario.description,
            )
        )

    if not isinstance(scenario.model_confidence, str):
        errors.append(
            ScenarioValidationError(
                field="model_confidence",
                message="model confidence must be a string",
                value=scenario.model_confidence,
            )
        )


def validate_scenario(
    scenario: ScenarioInput,
) -> ScenarioValidationResult:
    """
    Validate a complete wastewater treatment scenario.

    This performs scenario-domain validation before the scenario
    is passed into the existing V2.7 prediction/risk workflow.
    """

    errors: list[ScenarioValidationError] = []

    _validate_metadata(scenario, errors)
    _validate_wastewater(scenario, errors)
    _validate_process(scenario, errors)

    return ScenarioValidationResult(
        valid=not errors,
        errors=errors,
    )


def validate_optimization_constraint(
    constraint: OptimizationConstraint,
) -> ScenarioValidationResult:
    """
    Validate an optimization constraint.

    Optimization constraints describe allowable parameter ranges.
    """

    errors: list[ScenarioValidationError] = []

    if not constraint.parameter.strip():
        errors.append(
            ScenarioValidationError(
                field="parameter",
                message="parameter must not be empty",
                value=constraint.parameter,
            )
        )

    if (
        constraint.minimum is not None
        and (
            isinstance(constraint.minimum, bool)
            or not isinstance(constraint.minimum, (int, float))
        )
    ):
        errors.append(
            ScenarioValidationError(
                field="minimum",
                message="minimum must be numeric",
                value=constraint.minimum,
            )
        )

    if (
        constraint.maximum is not None
        and (
            isinstance(constraint.maximum, bool)
            or not isinstance(constraint.maximum, (int, float))
        )
    ):
        errors.append(
            ScenarioValidationError(
                field="maximum",
                message="maximum must be numeric",
                value=constraint.maximum,
            )
        )

    for field, value in (
        ("minimum", constraint.minimum),
        ("maximum", constraint.maximum),
    ):
        if value is None:
            continue

        if isinstance(value, bool) or not isinstance(
            value,
            (int, float),
        ):
            continue

        if not math.isfinite(float(value)):
            errors.append(
                ScenarioValidationError(
                    field=field,
                    message="value must be finite",
                    value=value,
                )
            )

    if (
        constraint.minimum is not None
        and constraint.maximum is not None
        and isinstance(constraint.minimum, (int, float))
        and not isinstance(constraint.minimum, bool)
        and isinstance(constraint.maximum, (int, float))
        and not isinstance(constraint.maximum, bool)
        and math.isfinite(float(constraint.minimum))
        and math.isfinite(float(constraint.maximum))
        and constraint.minimum > constraint.maximum
    ):
        errors.append(
            ScenarioValidationError(
                field="minimum",
                message="minimum must not be greater than maximum",
                value=constraint.minimum,
            )
        )

    return ScenarioValidationResult(
        valid=not errors,
        errors=errors,
    )


def validate_optimization_constraints(
    constraints: list[OptimizationConstraint],
) -> ScenarioValidationResult:
    """
    Validate a collection of optimization constraints.

    Duplicate parameters are rejected because a single parameter
    should have one unambiguous allowable range.
    """

    errors: list[ScenarioValidationError] = []

    seen_parameters: set[str] = set()

    for index, constraint in enumerate(constraints):
        result = validate_optimization_constraint(constraint)

        for error in result.errors:
            errors.append(
                ScenarioValidationError(
                    field=f"constraints[{index}].{error.field}",
                    message=error.message,
                    value=error.value,
                )
            )

        parameter = constraint.parameter.strip()

        if parameter in seen_parameters:
            errors.append(
                ScenarioValidationError(
                    field=f"constraints[{index}].parameter",
                    message="duplicate parameter constraint",
                    value=parameter,
                )
            )

        seen_parameters.add(parameter)

    return ScenarioValidationResult(
        valid=not errors,
        errors=errors,
    )