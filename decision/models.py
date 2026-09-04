from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class DecisionPriority(IntEnum):
    NORMAL = 0
    LOW = 1
    MODERATE = 2
    HIGH = 3
    CRITICAL = 4


@dataclass(frozen=True)
class DecisionInput:
    predicted_effluent_bod5: float
    anomaly_percentile: float
    overall_risk_level: int
    overall_risk_score: float
    dissolved_oxygen: float
    flow_m3_day: float
    hrt_hours: float
    influent_bod5: float
    influent_cod: float
    influent_tss: float
    model_confidence: str = "unknown"


@dataclass
class DecisionRecommendation:
    priority: DecisionPriority
    summary: str
    possible_contributors: list[str] = field(default_factory=list)
    checks_to_perform: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    monitoring_recommendations: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    @property
    def priority_name(self) -> str:
        return self.priority.name
