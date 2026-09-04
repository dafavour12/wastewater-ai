from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)

VALID_PROCESS_PAYLOAD = {
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


def test_process_anomaly_endpoint_exists():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200


def test_process_anomaly_response_structure():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200

    data = response.json()

    assert "anomaly_score" in data
    assert "anomaly_percentile" in data
    assert "is_anomaly" in data
    assert "risk_band" in data
    assert "alert_level" in data
    assert "message" in data
    assert "monitoring_method" in data
    assert "contamination" in data
    assert "features_used" in data


def test_process_anomaly_percentile_range():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200

    data = response.json()

    assert 0 <= data["anomaly_percentile"] <= 100


def test_process_anomaly_score_is_numeric():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["anomaly_score"], float)


def test_process_anomaly_boolean_flag():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200

    data = response.json()

    assert isinstance(data["is_anomaly"], bool)


def test_process_anomaly_risk_band_is_valid():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200

    data = response.json()

    assert data["risk_band"] in {
        "normal",
        "low",
        "elevated",
        "high",
        "critical",
    }


def test_process_anomaly_alert_level_is_valid():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200

    data = response.json()

    assert data["alert_level"] in {
        "normal",
        "watch",
        "alert",
    }


def test_process_anomaly_uses_22_features():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200

    data = response.json()

    assert data["features_used"] == 22


def test_process_anomaly_uses_isolation_forest():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200

    data = response.json()

    assert data["monitoring_method"] == "Isolation Forest"


def test_process_anomaly_uses_two_percent_contamination():
    response = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    assert response.status_code == 200

    data = response.json()

    assert data["contamination"] == 0.02


def test_process_anomaly_rejects_missing_feature():
    payload = VALID_PROCESS_PAYLOAD.copy()
    payload.pop("PH-P")
    response = client.post(
        "/process/anomaly",
        json=payload,
    )

    assert response.status_code == 422


def test_process_anomaly_rejects_invalid_feature():
    payload = VALID_PROCESS_PAYLOAD.copy()
    payload["DBO-P"] = "invalid"
    response = client.post(
        "/process/anomaly",
        json=payload,
    )

    assert response.status_code == 422


def test_process_anomaly_is_deterministic():
    response_1 = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )
    response_2 = client.post(
        "/process/anomaly",
        json=VALID_PROCESS_PAYLOAD,
    )

    assert response_1.status_code == 200
    assert response_2.status_code == 200

    data_1 = response_1.json()
    data_2 = response_2.json()

    assert data_1["anomaly_score"] == data_2["anomaly_score"]
    assert data_1["anomaly_percentile"] == data_2["anomaly_percentile"]
    assert data_1["is_anomaly"] == data_2["is_anomaly"]
    assert data_1["risk_band"] == data_2["risk_band"]
    assert data_1["alert_level"] == data_2["alert_level"]


def test_existing_predict_endpoint_still_works_after_monitoring_integration():
    response = client.post(
        "/predict",
        json={
            "influent_bod5": 300,
            "influent_cod": 570,
            "influent_tss": 250,
            "flow_m3_day": 1050,
            "dissolved_oxygen": 2.1,
            "temperature": 27,
            "hrt_hours": 8,
        },
    )
    assert response.status_code == 200

    data = response.json()

    assert "predicted_effluent_bod5" in data
    assert "status" in data
    assert "recommendation" in data
    assert "limitations" in data