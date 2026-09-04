from __future__ import annotations

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_WASTEWATER = {
    "influent_bod5": 300.0,
    "influent_cod": 570.0,
    "influent_tss": 250.0,
    "flow_m3_day": 1050.0,
    "dissolved_oxygen": 2.1,
    "temperature": 27.0,
    "hrt_hours": 8.0,
}


VALID_PROCESS = {
    "PH-P": 7.0,
    "DBO-P": 25.0,
    "SS-P": 30.0,
    "SSV-P": 20.0,
    "SED-P": 5.0,
    "COND-P": 500.0,
    "PH-D": 7.2,
    "DBO-D": 20.0,
    "DQO-D": 50.0,
    "SS-D": 25.0,
    "SSV-D": 18.0,
    "SED-D": 4.0,
    "COND-D": 480.0,
    "RD-DBO-P": 1.0,
    "RD-SS-P": 1.0,
    "RD-DBO-D": 1.0,
    "RD-SS-D": 1.0,
    "RD-DBO-G": 1.0,
    "RD-SS-G": 1.0,
    "RD-SED-G": 1.0,
    "RD-N-NH4": 1.0,
    "RD-N-NO2": 1.0,
}


def make_payload() -> dict:
    return {
        "wastewater": VALID_WASTEWATER.copy(),
        "process": VALID_PROCESS.copy(),
        "model_confidence": "research",
    }


def test_v27_end_to_end_decision_workflow():
    """
    Verify the complete V2.7 workflow from real model inference
    through risk assessment, decision generation, persistence,
    specific retrieval, and history.
    """

    create_response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    assert create_response.status_code == 200

    result = create_response.json()

    assessment_id = result["assessment_id"]

    assert isinstance(assessment_id, int)
    assert assessment_id > 0

    # BOD5 prediction reached the combined workflow.
    assert isinstance(
        result["predicted_effluent_bod5"],
        float,
    )
    assert result["predicted_effluent_bod5"] >= 0

    # Process anomaly monitoring reached the combined workflow.
    assert 0 <= result["anomaly_percentile"] <= 100
    assert isinstance(result["anomaly_score"], float)
    assert isinstance(result["is_anomaly"], bool)

    # Combined risk assessment was produced.
    assert result["overall_risk_level"] in {
        "normal",
        "low",
        "elevated",
        "high",
        "critical",
    }

    assert 0 <= result["overall_risk_score"] <= 100
    assert 0 <= result["prediction_score"] <= 100
    assert 0 <= result["confidence_score"] <= 100

    # V2.7 Decision Engine output was generated.
    assert result["decision_priority"] in {
        "normal",
        "low",
        "moderate",
        "high",
        "critical",
    }

    assert result["decision_summary"]

    for field in (
        "possible_contributors",
        "checks_to_perform",
        "recommended_actions",
        "monitoring_recommendations",
        "evidence",
        "limitations",
    ):
        assert isinstance(result[field], list)

    # Retrieve the exact persisted assessment.
    detail_response = client.get(
        f"/risk/assessments/{assessment_id}",
    )

    assert detail_response.status_code == 200

    detail = detail_response.json()

    assert detail["id"] == assessment_id

    # The decision persisted with the assessment.
    assert (
        detail["decision_priority"]
        == result["decision_priority"]
    )

    assert (
        detail["decision_summary"]
        == result["decision_summary"]
    )

    assert (
        detail["possible_contributors"]
        == result["possible_contributors"]
    )

    assert (
        detail["checks_to_perform"]
        == result["checks_to_perform"]
    )

    assert (
        detail["recommended_actions"]
        == result["recommended_actions"]
    )

    assert (
        detail["monitoring_recommendations"]
        == result["monitoring_recommendations"]
    )

    assert detail["evidence"] == result["evidence"]
    assert detail["limitations"] == result["limitations"]

    # Core risk outputs also survived persistence unchanged.
    assert (
        detail["predicted_effluent_bod5"]
        == result["predicted_effluent_bod5"]
    )

    assert (
        detail["anomaly_percentile"]
        == result["anomaly_percentile"]
    )

    assert (
        detail["overall_risk_level"]
        == result["overall_risk_level"]
    )

    assert (
        detail["overall_risk_score"]
        == result["overall_risk_score"]
    )

    # Finally verify the same assessment appears in history.
    history_response = client.get(
        "/risk/assessments",
    )

    assert history_response.status_code == 200

    history = history_response.json()

    matching = next(
        item
        for item in history
        if item["id"] == assessment_id
    )

    assert (
        matching["decision_priority"]
        == result["decision_priority"]
    )

    assert (
        matching["decision_summary"]
        == result["decision_summary"]
    )

    assert (
        matching["recommended_actions"]
        == result["recommended_actions"]
    )

    assert (
        matching["predicted_effluent_bod5"]
        == result["predicted_effluent_bod5"]
    )

    assert (
        matching["overall_risk_score"]
        == result["overall_risk_score"]
    )


def test_v27_end_to_end_decision_is_deterministic_for_same_input():
    """
    Verify that the same real combined input produces the same
    decision outputs across repeated workflow executions.
    """

    first_response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    second_response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    first = first_response.json()
    second = second_response.json()

    assert (
        first["predicted_effluent_bod5"]
        == second["predicted_effluent_bod5"]
    )

    assert (
        first["anomaly_score"]
        == second["anomaly_score"]
    )

    assert (
        first["anomaly_percentile"]
        == second["anomaly_percentile"]
    )

    assert (
        first["is_anomaly"]
        == second["is_anomaly"]
    )

    assert (
        first["overall_risk_level"]
        == second["overall_risk_level"]
    )

    assert (
        first["overall_risk_score"]
        == second["overall_risk_score"]
    )

    assert (
        first["decision_priority"]
        == second["decision_priority"]
    )

    assert (
        first["decision_summary"]
        == second["decision_summary"]
    )

    assert (
        first["possible_contributors"]
        == second["possible_contributors"]
    )

    assert (
        first["checks_to_perform"]
        == second["checks_to_perform"]
    )

    assert (
        first["recommended_actions"]
        == second["recommended_actions"]
    )

    assert (
        first["monitoring_recommendations"]
        == second["monitoring_recommendations"]
    )

    assert first["evidence"] == second["evidence"]
    assert first["limitations"] == second["limitations"]
