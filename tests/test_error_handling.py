from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_validation_error_has_standard_structure():
    response = client.post(
        "/predict",
        json={},
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] == "validation_error"
    assert data["message"] == "Request validation failed."
    assert data["status_code"] == 422
    assert "details" in data
    assert isinstance(data["details"], list)


def test_validation_error_rejects_negative_flow():
    response = client.post(
        "/predict",
        json={
            "influent_bod5": 280,
            "influent_cod": 520,
            "influent_tss": 240,
            "flow_m3_day": -1,
            "dissolved_oxygen": 2.0,
            "temperature": 27,
            "hrt_hours": 8,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] == "validation_error"
    assert data["status_code"] == 422


def test_validation_error_rejects_temperature_above_limit():
    response = client.post(
        "/predict",
        json={
            "influent_bod5": 280,
            "influent_cod": 520,
            "influent_tss": 240,
            "flow_m3_day": 1000,
            "dissolved_oxygen": 2.0,
            "temperature": 100,
            "hrt_hours": 8,
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] == "validation_error"


def test_not_found_prediction_returns_standard_error():
    response = client.get(
        "/predictions/999999999"
    )

    assert response.status_code == 404

    data = response.json()

    assert data["error"] == "http_error"
    assert data["message"] == "Prediction not found"
    assert data["status_code"] == 404


def test_not_found_risk_assessment_returns_standard_error():
    response = client.get(
        "/risk/assessments/999999999"
    )

    assert response.status_code == 404

    data = response.json()

    assert data["error"] == "http_error"
    assert data["message"] == "Risk assessment not found"
    assert data["status_code"] == 404


def test_invalid_risk_confidence_returns_standard_error():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 30,
            "anomaly_percentile": 50,
            "model_confidence": "invalid-confidence",
        },
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] == "http_error"
    assert data["status_code"] == 422
    assert "message" in data


def test_invalid_process_input_returns_validation_error():
    response = client.post(
        "/process/anomaly",
        json={},
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] == "validation_error"
    assert data["status_code"] == 422
    assert isinstance(data["details"], list)


def test_error_response_does_not_expose_internal_exception():
    response = client.get(
        "/predictions/999999999"
    )

    data = response.json()

    assert "Traceback" not in str(data)
    assert "FileNotFoundError" not in str(data)
    assert "sqlalchemy" not in str(data).lower()
