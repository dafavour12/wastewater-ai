import math

from fastapi.testclient import TestClient

from api.main import app
from api.guardrails import validate_prediction_guardrails


client = TestClient(app)


def valid_payload():
    return {
        "influent_bod5": 300.0,
        "influent_cod": 560.0,
        "influent_tss": 250.0,
        "flow_m3_day": 1050.0,
        "dissolved_oxygen": 2.5,
        "temperature": 25.0,
        "hrt_hours": 24.0,
    }


def test_predict_rejects_bod5_greater_than_cod():
    payload = valid_payload()
    payload["influent_bod5"] = 700.0
    payload["influent_cod"] = 500.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] == "http_error"
    assert data["status_code"] == 422

    assert any(
        item["field"] == "influent_bod5"
        for item in data["detail"]
    )


def test_predict_rejects_excessive_bod5():
    payload = valid_payload()
    payload["influent_bod5"] = 5001.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert any(
        item["field"] == "influent_bod5"
        for item in data["detail"]
    )


def test_predict_rejects_excessive_cod():
    payload = valid_payload()
    payload["influent_cod"] = 10001.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert any(
        item["field"] == "influent_cod"
        for item in data["detail"]
    )


def test_predict_rejects_excessive_dissolved_oxygen():
    payload = valid_payload()
    payload["dissolved_oxygen"] = 21.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert any(
        item["field"] == "dissolved_oxygen"
        for item in data["detail"]
    )


def test_predict_rejects_excessive_temperature():
    payload = valid_payload()
    payload["temperature"] = 61.0

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 422

    data = response.json()

    assert data["error"] == "validation_error"
    assert data["status_code"] == 422

    assert any(
        error["loc"][-1] == "temperature"
        for error in data["detail"]
    )


def test_guardrails_reject_nan_temperature():
    payload = valid_payload()
    payload["temperature"] = math.nan

    violations = validate_prediction_guardrails(payload)

    assert any(
        violation.field == "temperature"
        for violation in violations
    )


def test_guardrails_reject_infinite_flow():
    payload = valid_payload()
    payload["flow_m3_day"] = math.inf

    violations = validate_prediction_guardrails(payload)

    assert any(
        violation.field == "flow_m3_day"
        for violation in violations
    )


def test_predict_accepts_valid_payload():
    payload = valid_payload()

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "id" in data
    assert "predicted_effluent_bod5" in data
    assert data["unit"] == "mg/L"