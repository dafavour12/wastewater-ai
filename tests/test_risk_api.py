from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_risk_assess_normal_case():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 15,
            "anomaly_percentile": 50,
            "model_confidence": "high",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["risk_level"] == "normal"
    assert data["predicted_bod5"] == 15
    assert data["anomaly_percentile"] == 50
    assert data["prediction_score"] == 0
    assert data["anomaly_score"] == 50
    assert data["confidence_score"] == 100
    assert "risk_reason" in data
    assert "recommended_action" in data


def test_risk_assess_elevated_prediction():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 40,
            "anomaly_percentile": 80,
            "model_confidence": "high",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["risk_level"] == "elevated"
    assert data["prediction_score"] == 50
    assert data["anomaly_score"] == 80


def test_risk_assess_critical_anomaly():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 18,
            "anomaly_percentile": 99.8,
            "model_confidence": "high",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["risk_level"] == "critical"
    assert data["anomaly_score"] == 99.8


def test_risk_assess_combined_high_prediction_and_anomaly():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 40,
            "anomaly_percentile": 99.2,
            "model_confidence": "medium",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["risk_level"] == "critical"
    assert data["prediction_score"] == 50
    assert data["anomaly_score"] == 99.2
    assert data["confidence_score"] == 70


def test_risk_assess_accepts_moderate_confidence():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 25,
            "anomaly_percentile": 80,
            "model_confidence": "moderate",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["confidence_score"] == 70


def test_risk_assess_rejects_anomaly_above_100():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 20,
            "anomaly_percentile": 101,
            "model_confidence": "high",
        },
    )

    assert response.status_code == 422


def test_risk_assess_rejects_negative_anomaly():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 20,
            "anomaly_percentile": -1,
            "model_confidence": "high",
        },
    )

    assert response.status_code == 422


def test_risk_assess_rejects_negative_prediction():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": -1,
            "anomaly_percentile": 50,
            "model_confidence": "high",
        },
    )

    assert response.status_code == 422


def test_risk_assess_rejects_missing_confidence():
    response = client.post(
        "/risk/assess",
        json={
            "predicted_bod5": 20,
            "anomaly_percentile": 50,
        },
    )

    assert response.status_code == 422


def test_existing_predict_endpoint_still_exists():
    response = client.post(
        "/predict",
        json={
            "influent_bod5": 280,
            "influent_cod": 520,
            "influent_tss": 240,
            "flow_m3_day": 1000,
            "dissolved_oxygen": 2.0,
            "temperature": 27,
            "hrt_hours": 8,
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert "id" in data
    assert "predicted_effluent_bod5" in data
    assert "status" in data
    assert "recommendation" in data
    assert "limitations" in data
    