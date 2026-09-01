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