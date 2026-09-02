from fastapi.testclient import TestClient

from api.main import (
    app,
    classify_bod5,
    get_prediction_limitations,
    get_treatment_recommendation,
)


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Wastewater AI API is running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "wastewater-ai-api"
    assert data["version"] == "0.1.0"


def test_model_info():
    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert data["model_type"] == "RandomForestRegressor"
    assert data["target"] == "effluent_bod5"
    assert data["target_unit"] == "mg/L"


def test_model_info_features_and_metrics():
    response = client.get("/model")

    assert response.status_code == 200

    data = response.json()

    assert len(data["features"]) == 7

    assert "influent_bod5" in data["features"]
    assert "influent_cod" in data["features"]
    assert "influent_tss" in data["features"]
    assert "flow_m3_day" in data["features"]
    assert "dissolved_oxygen" in data["features"]
    assert "temperature" in data["features"]
    assert "hrt_hours" in data["features"]

    assert data["metrics"]["mae"] == 1.40
    assert data["metrics"]["rmse"] == 1.48
    assert data["metrics"]["r2"] == 0.93


def test_predict_valid_input():
    payload = {
        "influent_bod5": 300,
        "influent_cod": 570,
        "influent_tss": 250,
        "flow_m3_day": 1050,
        "dissolved_oxygen": 2.1,
        "temperature": 27,
        "hrt_hours": 8,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "id" in data
    assert "predicted_effluent_bod5" in data
    assert data["unit"] == "mg/L"

    assert "status" in data
    assert data["status"] in {
        "low",
        "moderate",
        "high",
        "very_high",
    }


def test_predict_invalid_input():
    payload = {
        "influent_bod5": 300,
        "influent_cod": 570,
        "influent_tss": 250,
        "flow_m3_day": 0,
        "dissolved_oxygen": 2.1,
        "temperature": 27,
        "hrt_hours": 0,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_predict_negative_temperature():
    payload = {
        "influent_bod5": 300,
        "influent_cod": 570,
        "influent_tss": 250,
        "flow_m3_day": 1050,
        "dissolved_oxygen": 2.1,
        "temperature": -5,
        "hrt_hours": 8,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422

    data = response.json()

    assert "detail" in data


def test_prediction_history():
    response = client.get("/predictions")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, list)


def test_get_prediction():
    payload = {
        "influent_bod5": 300,
        "influent_cod": 570,
        "influent_tss": 250,
        "flow_m3_day": 1050,
        "dissolved_oxygen": 2.1,
        "temperature": 27,
        "hrt_hours": 8,
    }

    create_response = client.post("/predict", json=payload)

    assert create_response.status_code == 200

    prediction_id = create_response.json()["id"]

    response = client.get(f"/predictions/{prediction_id}")

    assert response.status_code == 200

    data = response.json()

    assert data["id"] == prediction_id
    assert data["influent_bod5"] == 300
    assert data["predicted_effluent_bod5"] > 0
    assert data["status"] in {
        "low",
        "moderate",
        "high",
        "very_high",
    }


def test_get_prediction_not_found():
    response = client.get("/predictions/999999")

    assert response.status_code == 404

    data = response.json()

    assert data["detail"] == "Prediction not found"


def test_prediction_stats():
    response = client.get("/predictions/stats")

    assert response.status_code == 200

    data = response.json()

    assert "total_predictions" in data
    assert "average_predicted_bod5" in data
    assert "minimum_predicted_bod5" in data
    assert "maximum_predicted_bod5" in data


def test_prediction_stats_values():
    response = client.get("/predictions/stats")

    assert response.status_code == 200

    data = response.json()

    assert data["total_predictions"] >= 0
    assert data["average_predicted_bod5"] >= 0
    assert data["minimum_predicted_bod5"] >= 0
    assert data["maximum_predicted_bod5"] >= 0

    assert (
        data["minimum_predicted_bod5"]
        <= data["average_predicted_bod5"]
        <= data["maximum_predicted_bod5"]
    )


def test_classify_bod5():
    assert classify_bod5(10) == "low"
    assert classify_bod5(20) == "low"
    assert classify_bod5(25) == "moderate"
    assert classify_bod5(30) == "moderate"
    assert classify_bod5(40) == "high"
    assert classify_bod5(50) == "high"
    assert classify_bod5(60) == "very_high"


def test_treatment_recommendation():
    recommendation = get_treatment_recommendation(25)

    assert isinstance(recommendation, str)
    assert len(recommendation) > 0


def test_low_bod5_recommendation():
    recommendation = get_treatment_recommendation(15)

    assert "routine process monitoring" in recommendation.lower()


def test_high_bod5_recommendation():
    recommendation = get_treatment_recommendation(40)

    assert "organic loading" in recommendation.lower()
    assert "aeration" in recommendation.lower()


def test_prediction_recommendation_and_limitations():
    payload = {
        "influent_bod5": 300,
        "influent_cod": 570,
        "influent_tss": 250,
        "flow_m3_day": 1050,
        "dissolved_oxygen": 2.1,
        "temperature": 27,
        "hrt_hours": 8,
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "recommendation" in data
    assert isinstance(data["recommendation"], str)
    assert len(data["recommendation"]) > 0

    assert "limitations" in data
    assert isinstance(data["limitations"], str)
    assert "laboratory testing" in data["limitations"].lower()
