from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import api.main as main
from api.database.models import Base, RiskAssessmentRecord


def build_test_database():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )

    Base.metadata.create_all(bind=engine)

    TestSessionLocal = sessionmaker(
        bind=engine,
        autocommit=False,
        autoflush=False,
    )

    return engine, TestSessionLocal


def build_payload() -> dict:
    return {
        "wastewater": {
            "influent_bod5": 200.0,
            "influent_cod": 400.0,
            "influent_tss": 200.0,
            "flow_m3_day": 1000.0,
            "dissolved_oxygen": 2.0,
            "temperature": 25.0,
            "hrt_hours": 8.0,
        },
        "process": {
            "PH-P": 7.0,
            "DBO-P": 200.0,
            "SS-P": 200.0,
            "SSV-P": 100.0,
            "SED-P": 50.0,
            "COND-P": 500.0,
            "PH-D": 7.0,
            "DBO-D": 10.0,
            "DQO-D": 20.0,
            "SS-D": 10.0,
            "SSV-D": 5.0,
            "SED-D": 2.0,
            "COND-D": 500.0,
            "RD-DBO-P": 1.0,
            "RD-SS-P": 1.0,
            "RD-DBO-D": 1.0,
            "RD-SS-D": 1.0,
            "RD-DBO-G": 1.0,
            "RD-SS-G": 1.0,
            "RD-SED-G": 1.0,
            "RD-N-NH4": 1.0,
            "RD-N-NO2": 1.0,
        },
        "model_confidence": "high",
    }


class FakeBOD5Model:
    def predict(self, dataframe):
        return [15.0]


class FakeProcessMonitor:
    contamination = 0.02

    def predict(self, process_data):
        return SimpleNamespace(
            anomaly_score=0.10,
            anomaly_percentile=20.0,
            is_anomaly=False,
            risk_band="normal",
            alert_level="none",
            message=(
                "Process conditions are within "
                "the expected range."
            ),
        )


def test_combined_risk_persists_decision_fields():
    engine, TestSessionLocal = build_test_database()

    def override_get_db():
        db = TestSessionLocal()

        try:
            yield db

        finally:
            db.close()

    original_model = main.model
    original_process_monitor = main.process_monitor

    main.model = FakeBOD5Model()
    main.process_monitor = FakeProcessMonitor()

    main.app.dependency_overrides[
        main.get_db
    ] = override_get_db

    try:
        client = TestClient(main.app)

        response = client.post(
            "/risk/assess/process",
            json=build_payload(),
        )

        assert response.status_code == 200

        body = response.json()

        assessment_id = body["assessment_id"]

        # ---------------------------------------------------------------
        # Verify Decision Engine fields are returned by the API.
        # ---------------------------------------------------------------

        assert body["decision_priority"] is not None

        assert body["decision_summary"] is not None

        assert isinstance(
            body["possible_contributors"],
            list,
        )

        assert isinstance(
            body["checks_to_perform"],
            list,
        )

        assert isinstance(
            body["recommended_actions"],
            list,
        )

        assert isinstance(
            body["monitoring_recommendations"],
            list,
        )

        assert isinstance(
            body["evidence"],
            list,
        )

        assert isinstance(
            body["limitations"],
            list,
        )

        # ---------------------------------------------------------------
        # Verify the Decision Engine fields were persisted.
        # ---------------------------------------------------------------

        db = TestSessionLocal()

        try:
            record = (
                db.query(RiskAssessmentRecord)
                .filter(
                    RiskAssessmentRecord.id
                    == assessment_id
                )
                .first()
            )

            assert record is not None

            assert record.decision_priority is not None

            assert record.decision_summary is not None

            assert isinstance(
                record.possible_contributors,
                list,
            )

            assert isinstance(
                record.checks_to_perform,
                list,
            )

            assert isinstance(
                record.recommended_actions,
                list,
            )

            assert isinstance(
                record.monitoring_recommendations,
                list,
            )

            assert isinstance(
                record.evidence,
                list,
            )

            assert isinstance(
                record.limitations,
                list,
            )

        finally:
            db.close()

    finally:
        main.model = original_model
        main.process_monitor = original_process_monitor

        main.app.dependency_overrides.clear()

        engine.dispose()


def test_combined_risk_history_returns_decision_fields():
    engine, TestSessionLocal = build_test_database()

    def override_get_db():
        db = TestSessionLocal()

        try:
            yield db

        finally:
            db.close()

    original_model = main.model
    original_process_monitor = main.process_monitor

    main.model = FakeBOD5Model()
    main.process_monitor = FakeProcessMonitor()

    main.app.dependency_overrides[
        main.get_db
    ] = override_get_db

    try:
        client = TestClient(main.app)

        # ---------------------------------------------------------------
        # First create a combined risk assessment.
        # ---------------------------------------------------------------

        create_response = client.post(
            "/risk/assess/process",
            json=build_payload(),
        )

        assert create_response.status_code == 200

        assessment_id = (
            create_response.json()["assessment_id"]
        )

        # ---------------------------------------------------------------
        # Retrieve risk assessment history.
        # ---------------------------------------------------------------

        history_response = client.get(
            "/risk/assessments"
        )

        assert history_response.status_code == 200

        history = history_response.json()

        assert isinstance(history, list)

        assert len(history) >= 1

        matching_record = next(
            item
            for item in history
            if item["id"] == assessment_id
        )

        # ---------------------------------------------------------------
        # Verify all V2.7 Decision Engine fields are exposed
        # through the history endpoint.
        # ---------------------------------------------------------------

        assert "decision_priority" in matching_record

        assert "decision_summary" in matching_record

        assert "possible_contributors" in matching_record

        assert "checks_to_perform" in matching_record

        assert "recommended_actions" in matching_record

        assert (
            "monitoring_recommendations"
            in matching_record
        )

        assert "evidence" in matching_record

        assert "limitations" in matching_record

        assert (
            matching_record["decision_priority"]
            is not None
        )

        assert (
            matching_record["decision_summary"]
            is not None
        )

        assert isinstance(
            matching_record["possible_contributors"],
            list,
        )

        assert isinstance(
            matching_record["checks_to_perform"],
            list,
        )

        assert isinstance(
            matching_record["recommended_actions"],
            list,
        )

        assert isinstance(
            matching_record[
                "monitoring_recommendations"
            ],
            list,
        )

        assert isinstance(
            matching_record["evidence"],
            list,
        )

        assert isinstance(
            matching_record["limitations"],
            list,
        )

    finally:
        main.model = original_model
        main.process_monitor = original_process_monitor

        main.app.dependency_overrides.clear()

        engine.dispose()
