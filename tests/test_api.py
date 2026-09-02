from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200

    data = response.json()

    assert data["message"] == "Wastewater AI API is running"


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