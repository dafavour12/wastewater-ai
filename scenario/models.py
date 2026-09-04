from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class ScenarioStatus(IntEnum):
    """
    Lifecycle status of a scenario.
    """

    CREATED = 0
    ANALYZED = 1
    COMPARED = 2
    OPTIMIZED = 3


@dataclass(frozen=True)
class ScenarioInput:
    """
    Defines the inputs for one wastewater treatment scenario.

    A scenario describes a specific set of wastewater and process
    conditions that will later be evaluated by the existing V2.7
    prediction, anomaly, risk, and decision engines.
    """

    name: str
    description: str

    wastewater: dict[str, float]
    process: dict[str, float]

    model_confidence: str = "unknown"

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OptimizationConstraint:
    """
    Defines an allowable range for one scenario variable.

    A constraint answers:

        "What values are acceptable?"

    It does not represent an actual scenario value.
    """

    parameter: str
    minimum: float | None = None
    maximum: float | None = None
    unit: str | None = None

    def __post_init__(self) -> None:
        if not self.parameter.strip():
            raise ValueError("parameter must not be empty")

        if self.minimum is None and self.maximum is None:
            raise ValueError(
                "at least one of minimum or maximum must be provided"
            )

        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError(
                "minimum must not be greater than maximum"
            )


@dataclass
class ScenarioResult:
    """
    Stores the engineering results produced after a scenario
    has been evaluated.

    The scenario layer stores results from the existing V2.7
    engines rather than reimplementing those calculations.
    """

    scenario_name: str

    predicted_effluent_bod5: float

    anomaly_score: float
    anomaly_percentile: float

    overall_risk_level: str
    overall_risk_score: float

    decision_priority: str
    decision_summary: str

    recommended_actions: list[str] = field(default_factory=list)
    monitoring_recommendations: list[str] = field(
        default_factory=list
    )

    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    status: ScenarioStatus = ScenarioStatus.ANALYZED

    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioComparison:
    """
    Represents the comparison of multiple evaluated scenarios.
    """

    scenarios: list[ScenarioResult] = field(default_factory=list)

    preferred_scenario: str | None = None

    selection_reason: str | None = None

    trade_offs: list[str] = field(default_factory=list)

    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def scenario_count(self) -> int:
        """
        Return the number of scenarios included in the comparison.
        """

        return len(self.scenarios)
