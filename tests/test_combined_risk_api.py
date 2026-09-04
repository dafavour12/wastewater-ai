from fastapi.testclient import TestClient
from copy import deepcopy
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


def test_combined_risk_endpoint_exists():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    assert response.status_code == 200


def test_combined_response_contains_prediction():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert "predicted_effluent_bod5" in data
    assert isinstance(
        data["predicted_effluent_bod5"],
        float,
    )


def test_combined_response_contains_anomaly_result():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert "anomaly_score" in data
    assert "anomaly_percentile" in data
    assert "is_anomaly" in data
    assert "anomaly_risk_band" in data
    assert "anomaly_alert_level" in data


def test_combined_response_contains_risk_result():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert "overall_risk_level" in data
    assert "overall_risk_score" in data
    assert "prediction_score" in data
    assert "confidence_score" in data
    assert "risk_reason" in data
    assert "recommended_action" in data


def test_anomaly_percentile_is_generated_by_process_engine():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert 0 <= data["anomaly_percentile"] <= 100


def test_risk_score_is_valid():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert 0 <= data["overall_risk_score"] <= 100


def test_prediction_score_is_valid():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert 0 <= data["prediction_score"] <= 100


def test_confidence_score_is_valid():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert 0 <= data["confidence_score"] <= 100


def test_monitoring_method_is_isolation_forest():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert data["monitoring_method"] == "Isolation Forest"


def test_process_feature_count_is_22():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert data["process_features_used"] == 22


def test_contamination_is_two_percent():
    response = client.post(
        "/risk/assess/process",
        json=make_payload(),
    )

    data = response.json()

    assert data["contamination"] == 0.02


def test_invalid_confidence_is_rejected():
    payload = make_payload()
    payload["model_confidence"] = "invalid"

    response = client.post(
        "/risk/assess/process",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_process_variable_is_rejected():
    payload = make_payload()
    del payload["process"]["PH-P"]

    response = client.post(
        "/risk/assess/process",
        json=payload,
    )

    assert response.status_code == 422


def test_negative_process_value_is_rejected():
    payload = make_payload()
    payload["process"]["PH-P"] = -1

    response = client.post(
        "/risk/assess/process",
        json=payload,
    )

    assert response.status_code == 422


def test_missing_wastewater_input_is_rejected():
    payload = make_payload()
    del payload["wastewater"]["influent_bod5"]

    response = client.post(
        "/risk/assess/process",
        json=payload,
    )

    assert response.status_code == 422


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


def test_combined_endpoint_is_deterministic():
    payload = make_payload()

    first = client.post(
        "/risk/assess/process",
        json=payload,
    )

    second = client.post(
        "/risk/assess/process",
        json=payload,
    )

    assert first.status_code == 200
    assert second.status_code == 200

    first_data = first.json()
    second_data = second.json()

    assert (
        first_data["predicted_effluent_bod5"]
        == second_data["predicted_effluent_bod5"]
    )

    assert (
        first_data["anomaly_percentile"]
        == second_data["anomaly_percentile"]
    )

    assert (
        first_data["overall_risk_level"]
        == second_data["overall_risk_level"]
    )

    assert (
        first_data["overall_risk_score"]
        == second_data["overall_risk_score"]
    )
