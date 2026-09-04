from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_health_endpoint_returns_healthy():
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"
    assert data["service"] == "wastewater-ai-api"
    assert data["version"] == "0.1.0"


def test_status_endpoint_exists():
    response = client.get("/status")

    assert response.status_code == 200


def test_status_endpoint_contains_required_fields():
    response = client.get("/status")

    data = response.json()

    assert "status" in data
    assert "service" in data
    assert "version" in data
    assert "environment" in data
    assert "models_ready" in data


def test_status_endpoint_reports_models_ready():
    response = client.get("/status")

    data = response.json()

    assert data["status"] == "ready"
    assert data["models_ready"] is True


def test_models_status_endpoint_exists():
    response = client.get("/models/status")

    assert response.status_code == 200


def test_models_status_contains_required_fields():
    response = client.get("/models/status")

    data = response.json()

    assert "status" in data
    assert "models" in data

    assert isinstance(
        data["models"],
        list,
    )


def test_models_status_contains_bod5_model():
    response = client.get("/models/status")

    data = response.json()

    bod5_models = [
        model
        for model in data["models"]
        if model["name"] == "BOD5 prediction model"
    ]

    assert len(bod5_models) == 1

    bod5_model = bod5_models[0]

    assert bod5_model["available"] is True
    assert (
        bod5_model["model_type"]
        == "RandomForestRegressor"
    )
    assert bod5_model["path"]


def test_models_status_contains_process_model():
    response = client.get("/models/status")

    data = response.json()

    process_models = [
        model
        for model in data["models"]
        if model["name"]
        == "Process anomaly model"
    ]

    assert len(process_models) == 1

    process_model = process_models[0]

    assert process_model["available"] is True
    assert (
        process_model["model_type"]
        == "IsolationForest"
    )
    assert process_model["path"]


def test_models_status_reports_ready():
    response = client.get("/models/status")

    data = response.json()

    assert data["status"] == "ready"

    for model in data["models"]:
        assert model["available"] is True
