from copy import deepcopy

from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


VALID_WASTEWATER = {
    "influent_bod5": 300,
    "influent_cod": 560,
    "influent_tss": 250,
    "flow_m3_day": 1050,
    "dissolved_oxygen": 2.1,
    "temperature": 27,
    "hrt_hours": 8,
}


VALID_PROCESS = {
    "PH-P": 7.0,
    "DBO-P": 200.0,
    "SS-P": 150.0,
    "SSV-P": 100.0,
    "SED-P": 20.0,
    "COND-P": 500.0,
    "PH-D": 7.1,
    "DBO-D": 100.0,
    "DQO-D": 200.0,
    "SS-D": 80.0,
    "SSV-D": 50.0,
    "SED-D": 10.0,
    "COND-D": 450.0,
    "RD-DBO-P": 0.5,
    "RD-SS-P": 0.4,
    "RD-DBO-D": 0.5,
    "RD-SS-D": 0.5,
    "RD-DBO-G": 0.2,
    "RD-SS-G": 0.2,
    "RD-SED-G": 0.1,
    "RD-N-NH4": 0.3,
    "RD-N-NO2": 0.1,
}


def make_payload():
    return {
        "wastewater": deepcopy(VALID_WASTEWATER),
        "process": deepcopy(VALID_PROCESS),
        "model_confidence": "research",
    }


def test_combined_risk_creates_database_record():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "assessment_id" in data
    assert isinstance(
        data["assessment_id"],
        int,
    )
    assert data["assessment_id"] > 0


def test_risk_history_endpoint_exists():
    response = client.get(
        "/risk/assessments",
    )

    assert response.status_code == 200
    assert isinstance(response.json(), list)


def test_risk_history_contains_required_fields():
    create_response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    assert create_response.status_code == 200

    assessment_id = create_response.json()[
        "assessment_id"
    ]

    response = client.get(
        f"/risk/assessments/{assessment_id}",
    )

    assert response.status_code == 200

    data = response.json()

    required_fields = [
        "id",
        "predicted_effluent_bod5",
        "prediction_status",
        "anomaly_score",
        "anomaly_percentile",
        "is_anomaly",
        "anomaly_risk_band",
        "anomaly_alert_level",
        "overall_risk_level",
        "overall_risk_score",
        "prediction_score",
        "confidence_score",
        "model_confidence",
        "risk_reason",
        "recommended_action",
        "monitoring_method",
        "contamination",
        "process_features_used",
        "created_at",
    ]

    for field in required_fields:
        assert field in data


def test_risk_history_returns_created_assessment():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    assessment_id = response.json()[
        "assessment_id"
    ]

    history_response = client.get(
        "/risk/assessments",
    )

    assert history_response.status_code == 200

    assessments = history_response.json()

    ids = [
        assessment["id"]
        for assessment in assessments
    ]

    assert assessment_id in ids


def test_get_specific_risk_assessment():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    assessment_id = response.json()[
        "assessment_id"
    ]

    response = client.get(
        f"/risk/assessments/{assessment_id}",
    )

    assert response.status_code == 200
    assert response.json()["id"] == assessment_id


def test_missing_risk_assessment_returns_404():
    response = client.get(
        "/risk/assessments/999999999",
    )

    assert response.status_code == 404


def test_risk_statistics_endpoint_exists():
    response = client.get(
        "/risk/assessments/stats",
    )

    assert response.status_code == 200


def test_risk_statistics_contains_required_fields():
    response = client.get(
        "/risk/assessments/stats",
    )

    data = response.json()

    assert "total_assessments" in data
    assert "average_risk_score" in data
    assert "minimum_risk_score" in data
    assert "maximum_risk_score" in data
    assert "risk_level_counts" in data
    assert "anomaly_count" in data


def test_risk_statistics_values_are_valid():
    response = client.get(
        "/risk/assessments/stats",
    )

    data = response.json()

    assert data["total_assessments"] >= 0
    assert data["average_risk_score"] >= 0
    assert data["minimum_risk_score"] >= 0
    assert data["maximum_risk_score"] >= 0
    assert data["anomaly_count"] >= 0


def test_risk_statistics_count_matches_history():
    history_response = client.get(
        "/risk/assessments",
    )

    stats_response = client.get(
        "/risk/assessments/stats",
    )

    history = history_response.json()
    stats = stats_response.json()

    assert stats["total_assessments"] == len(history)


def test_risk_statistics_risk_levels_sum_to_total():
    response = client.get(
        "/risk/assessments/stats",
    )

    data = response.json()

    counts = data["risk_level_counts"]

    assert sum(counts.values()) == (
        data["total_assessments"]
    )


def test_persisted_values_match_combined_response():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    assert response.status_code == 200

    result = response.json()
    assessment_id = result["assessment_id"]

    history_response = client.get(
        f"/risk/assessments/{assessment_id}",
    )

    assert history_response.status_code == 200

    stored = history_response.json()

    assert (
        stored["predicted_effluent_bod5"]
        == result["predicted_effluent_bod5"]
    )

    assert (
        stored["anomaly_percentile"]
        == result["anomaly_percentile"]
    )

    assert (
        stored["overall_risk_level"]
        == result["overall_risk_level"]
    )

    assert (
        stored["overall_risk_score"]
        == result["overall_risk_score"]
    )


def test_existing_risk_endpoint_still_works():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 25,
            "anomaly_percentile": 50,
            "model_confidence": "high",
        },
    )

    assert response.status_code == 200


def test_existing_process_anomaly_endpoint_still_works():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS,
    )

    assert response.status_code == 200
