from datetime import UTC, datetime

from sqlalchemy import create_engine, inspect, text

from api.database.migrations import (
    migrate_risk_assessment_decision_fields,
)


LEGACY_RISK_ASSESSMENTS_SQL = """
CREATE TABLE risk_assessments (
    id INTEGER PRIMARY KEY,
    predicted_effluent_bod5 FLOAT NOT NULL,
    prediction_status VARCHAR(20) NOT NULL,
    anomaly_score FLOAT NOT NULL,
    anomaly_percentile FLOAT NOT NULL,
    is_anomaly BOOLEAN NOT NULL,
    anomaly_risk_band VARCHAR(20) NOT NULL,
    anomaly_alert_level VARCHAR(20) NOT NULL,
    overall_risk_level VARCHAR(20) NOT NULL,
    overall_risk_score FLOAT NOT NULL,
    prediction_score FLOAT NOT NULL,
    confidence_score FLOAT NOT NULL,
    model_confidence VARCHAR(20) NOT NULL,
    risk_reason VARCHAR(500) NOT NULL,
    recommended_action VARCHAR(500) NOT NULL,
    monitoring_method VARCHAR(100) NOT NULL,
    contamination FLOAT NOT NULL,
    process_features_used INTEGER NOT NULL,
    created_at DATETIME NOT NULL
)
"""


def create_legacy_database():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )

    with engine.begin() as connection:
        connection.execute(
            text(LEGACY_RISK_ASSESSMENTS_SQL)
        )

    return engine


def insert_legacy_risk_assessment(engine):
    with engine.begin() as connection:
        connection.execute(
            text(
                """
                INSERT INTO risk_assessments (
                    predicted_effluent_bod5,
                    prediction_status,
                    anomaly_score,
                    anomaly_percentile,
                    is_anomaly,
                    anomaly_risk_band,
                    anomaly_alert_level,
                    overall_risk_level,
                    overall_risk_score,
                    prediction_score,
                    confidence_score,
                    model_confidence,
                    risk_reason,
                    recommended_action,
                    monitoring_method,
                    contamination,
                    process_features_used,
                    created_at
                )
                VALUES (
                    10.0,
                    'success',
                    0.1,
                    20.0,
                    0,
                    'normal',
                    'normal',
                    'normal',
                    0.5,
                    0.2,
                    100.0,
                    'high',
                    'Normal operation.',
                    'Continue monitoring.',
                    'Routine monitoring.',
                    0.02,
                    22,
                    :created_at
                )
                """
            ),
            {
                "created_at": datetime.now(UTC),
            },
        )


def test_migration_adds_decision_columns():
    engine = create_legacy_database()

    columns_before = {
        column["name"]
        for column in inspect(engine).get_columns(
            "risk_assessments"
        )
    }

    assert "decision_priority" not in columns_before
    assert "decision_summary" not in columns_before
    assert "possible_contributors" not in columns_before
    assert "checks_to_perform" not in columns_before
    assert "recommended_actions" not in columns_before
    assert "monitoring_recommendations" not in columns_before
    assert "evidence" not in columns_before
    assert "limitations" not in columns_before

    added = migrate_risk_assessment_decision_fields(engine)

    assert set(added) == {
        "decision_priority",
        "decision_summary",
        "possible_contributors",
        "checks_to_perform",
        "recommended_actions",
        "monitoring_recommendations",
        "evidence",
        "limitations",
    }

    columns_after = {
        column["name"]
        for column in inspect(engine).get_columns(
            "risk_assessments"
        )
    }

    assert "decision_priority" in columns_after
    assert "decision_summary" in columns_after
    assert "possible_contributors" in columns_after
    assert "checks_to_perform" in columns_after
    assert "recommended_actions" in columns_after
    assert "monitoring_recommendations" in columns_after
    assert "evidence" in columns_after
    assert "limitations" in columns_after


def test_migration_is_idempotent():
    engine = create_legacy_database()

    first_run = migrate_risk_assessment_decision_fields(engine)
    second_run = migrate_risk_assessment_decision_fields(engine)

    assert len(first_run) == 8
    assert second_run == []


def test_existing_risk_assessment_record_is_preserved():
    engine = create_legacy_database()

    insert_legacy_risk_assessment(engine)

    migrate_risk_assessment_decision_fields(engine)

    with engine.connect() as connection:
        row = connection.execute(
            text(
                """
                SELECT
                    id,
                    predicted_effluent_bod5,
                    overall_risk_level,
                    overall_risk_score,
                    model_confidence,
                    decision_priority,
                    decision_summary,
                    possible_contributors
                FROM risk_assessments
                """
            )
        ).mappings().one()

    assert row["id"] == 1
    assert row["predicted_effluent_bod5"] == 10.0
    assert row["overall_risk_level"] == "normal"
    assert row["overall_risk_score"] == 0.5
    assert row["model_confidence"] == "high"

    assert row["decision_priority"] is None
    assert row["decision_summary"] is None
    assert row["possible_contributors"] is None
