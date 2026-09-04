from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class GuardrailViolation:
    field: str
    value: object
    message: str


# ---------------------------------------------------------------------------
# Wastewater prediction input limits
# ---------------------------------------------------------------------------
#
# These are engineering sanity limits, not treatment-design code limits.
# They are deliberately broad enough to avoid rejecting unusual but possible
# wastewater conditions while still catching obvious bad API inputs.
#
# The limits are intended as application-level data-quality guardrails.
# They must not be interpreted as regulatory discharge limits.
# ---------------------------------------------------------------------------

PREDICTION_LIMITS: dict[str, tuple[float, float]] = {
    "influent_bod5": (0.0, 5000.0),
    "influent_cod": (0.0, 10000.0),
    "influent_tss": (0.0, 10000.0),
    "flow_m3_day": (0.0, 1_000_000.0),
    "dissolved_oxygen": (0.0, 20.0),
    "temperature": (-5.0, 60.0),
    "hrt_hours": (0.0, 1000.0),
}


def is_finite_number(value: object) -> bool:
    """Return True when value can be interpreted as a finite number."""
    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(numeric_value)


def validate_finite_value(
    field: str,
    value: object,
) -> GuardrailViolation | None:
    """Reject NaN, positive infinity, and negative infinity."""
    if not is_finite_number(value):
        return GuardrailViolation(
            field=field,
            value=value,
            message=f"{field} must be a finite numeric value.",
        )

    return None


def validate_range(
    field: str,
    value: object,
    minimum: float,
    maximum: float,
) -> GuardrailViolation | None:
    """Validate a numeric field against a broad engineering sanity range."""

    finite_violation = validate_finite_value(field, value)

    if finite_violation is not None:
        return finite_violation

    numeric_value = float(value)

    if numeric_value < minimum or numeric_value > maximum:
        return GuardrailViolation(
            field=field,
            value=value,
            message=(
                f"{field} must be between "
                f"{minimum:g} and {maximum:g}."
            ),
        )

    return None


def validate_prediction_input(
    data: dict[str, object],
) -> list[GuardrailViolation]:
    """
    Validate the complete wastewater prediction input.

    Returns every detected violation rather than stopping at the first one.
    This allows the API to provide useful feedback for multiple bad fields.
    """

    violations: list[GuardrailViolation] = []

    for field, (minimum, maximum) in PREDICTION_LIMITS.items():
        if field not in data:
            violations.append(
                GuardrailViolation(
                    field=field,
                    value=None,
                    message=f"{field} is required.",
                )
            )
            continue

        violation = validate_range(
            field=field,
            value=data[field],
            minimum=minimum,
            maximum=maximum,
        )

        if violation is not None:
            violations.append(violation)

    return violations


def validate_prediction_relationships(
    data: dict[str, object],
) -> list[GuardrailViolation]:
    """
    Validate relationships between wastewater process variables.

    These checks are intentionally conservative. They identify values that
    are internally suspicious without pretending to replace engineering
    judgement or a treatment-process model.
    """

    violations: list[GuardrailViolation] = []

    required_fields = (
        "influent_bod5",
        "influent_cod",
        "influent_tss",
        "flow_m3_day",
        "dissolved_oxygen",
        "temperature",
        "hrt_hours",
    )

    if any(field not in data for field in required_fields):
        return violations

    if not all(
        is_finite_number(data[field])
        for field in required_fields
    ):
        return violations

    bod5 = float(data["influent_bod5"])
    cod = float(data["influent_cod"])
    flow = float(data["flow_m3_day"])
    hrt = float(data["hrt_hours"])

    # COD is normally expected to be at least as large as BOD5.
    if bod5 > cod:
        violations.append(
            GuardrailViolation(
                field="influent_bod5",
                value=bod5,
                message=(
                    "influent_bod5 should not normally exceed "
                    "influent_cod."
                ),
            )
        )

    # A positive flow should accompany a positive hydraulic retention time.
    if flow == 0.0 and hrt > 0.0:
        violations.append(
            GuardrailViolation(
                field="flow_m3_day",
                value=flow,
                message=(
                    "flow_m3_day cannot be zero when hrt_hours "
                    "is greater than zero."
                ),
            )
        )

    return violations


def validate_prediction_guardrails(
    data: dict[str, object],
) -> list[GuardrailViolation]:
    """
    Run all prediction input guardrails.

    This is the main entry point intended for API integration.
    """

    violations = validate_prediction_input(data)

    # Relationship checks are only useful when the basic fields are valid.
    if not violations:
        violations.extend(
            validate_prediction_relationships(data)
        )

    return violations


def guardrails_pass(
    data: dict[str, object],
) -> bool:
    """Return True when all configured guardrails pass."""
    return not validate_prediction_guardrails(data)
