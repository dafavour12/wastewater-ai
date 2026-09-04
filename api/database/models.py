from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import DateTime, Float, Integer, JSON, String
from sqlalchemy.orm import Mapped, mapped_column

from .database import Base


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    influent_bod5: Mapped[float] = mapped_column(Float)
    influent_cod: Mapped[float] = mapped_column(Float)
    influent_tss: Mapped[float] = mapped_column(Float)
    flow_m3_day: Mapped[float] = mapped_column(Float)
    dissolved_oxygen: Mapped[float] = mapped_column(Float)
    temperature: Mapped[float] = mapped_column(Float)
    hrt_hours: Mapped[float] = mapped_column(Float)

    predicted_effluent_bod5: Mapped[float] = mapped_column(Float)

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
    )


class RiskAssessmentRecord(Base):
    __tablename__ = "risk_assessments"

    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        index=True,
    )

    predicted_effluent_bod5: Mapped[float] = mapped_column(
        Float
    )

    prediction_status: Mapped[str] = mapped_column(
        String(20)
    )

    anomaly_score: Mapped[float] = mapped_column(
        Float
    )

    anomaly_percentile: Mapped[float] = mapped_column(
        Float
    )

    is_anomaly: Mapped[bool] = mapped_column()

    anomaly_risk_band: Mapped[str] = mapped_column(
        String(20)
    )

    anomaly_alert_level: Mapped[str] = mapped_column(
        String(20)
    )

    overall_risk_level: Mapped[str] = mapped_column(
        String(20)
    )

    overall_risk_score: Mapped[float] = mapped_column(
        Float
    )

    prediction_score: Mapped[float] = mapped_column(
        Float
    )

    confidence_score: Mapped[float] = mapped_column(
        Float
    )

    model_confidence: Mapped[str] = mapped_column(
        String(20)
    )

    risk_reason: Mapped[str] = mapped_column(
        String(500)
    )

    recommended_action: Mapped[str] = mapped_column(
        String(500)
    )

    monitoring_method: Mapped[str] = mapped_column(
        String(100)
    )

    contamination: Mapped[float] = mapped_column(
        Float
    )

    process_features_used: Mapped[int] = mapped_column(
        Integer
    )

    # ---------------------------------------------------------
    # V2.7 Decision Engine persistence
    # ---------------------------------------------------------
    #
    # These fields are nullable because existing records in
    # wastewater.db were created before the Decision Engine
    # existed.
    #

    decision_priority: Mapped[str | None] = mapped_column(
        String(20),
        nullable=True,
    )

    decision_summary: Mapped[str | None] = mapped_column(
        String(1000),
        nullable=True,
    )

    possible_contributors: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    checks_to_perform: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    recommended_actions: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    monitoring_recommendations: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    evidence: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    limitations: Mapped[list[str] | None] = mapped_column(
        JSON,
        nullable=True,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=lambda: datetime.now(UTC),
    )
