import math

from fastapi.testclient import TestClient

import api.main as main
from api.database.database import SessionLocal
from api.database.models import Prediction


client = TestClient(main.app)


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


def prediction_count():
    db = SessionLocal()

    try:
        return db.query(Prediction).count()
    finally:
        db.close()


def test_status_is_ready_when_models_are_available(monkeypatch):
    monkeypatch.setattr(
        main,
        "model",
        object(),
    )

    monkeypatch.setattr(
        main,
        "process_monitor",
        object(),
    )

    response = client.get("/status")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "ready"
    assert data["models_ready"] is True


def test_status_is_degraded_when_bod5_model_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        main,
        "model",
        None,
    )

    monkeypatch.setattr(
        main,
        "process_monitor",
        object(),
    )

    response = client.get("/status")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "degraded"
    assert data["models_ready"] is False


def test_models_status_reports_unavailable_bod5_model(
    monkeypatch,
):
    monkeypatch.setattr(
        main,
        "model",
        None,
    )

    monkeypatch.setattr(
        main,
        "process_monitor",
        object(),
    )

    response = client.get("/models/status")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "degraded"

    bod5_model = next(
        item
        for item in data["models"]
        if item["name"] == "BOD5 prediction model"
    )

    assert bod5_model["available"] is False


def test_predict_returns_503_when_model_is_unavailable(
    monkeypatch,
):
    monkeypatch.setattr(
        main,
        "model",
        None,
    )

    before = prediction_count()

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    after = prediction_count()

    assert response.status_code == 503

    data = response.json()

    assert data["error"] == "http_error"
    assert (
        data["message"]
        == "BOD5 prediction model is currently unavailable."
    )

    assert after == before


def test_predict_returns_503_when_model_inference_fails(
    monkeypatch,
):
    class BrokenModel:
        def predict(self, features):
            raise RuntimeError(
                "simulated model failure"
            )

    monkeypatch.setattr(
        main,
        "model",
        BrokenModel(),
    )

    before = prediction_count()

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    after = prediction_count()

    assert response.status_code == 503

    data = response.json()

    assert data["error"] == "http_error"
    assert (
        data["message"]
        == (
            "BOD5 prediction could not be completed because "
            "the prediction model failed during inference."
        )
    )

    assert after == before


def test_predict_returns_503_when_model_returns_nan(
    monkeypatch,
):
    class NaNModel:
        def predict(self, features):
            return [math.nan]

    monkeypatch.setattr(
        main,
        "model",
        NaNModel(),
    )

    before = prediction_count()

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    after = prediction_count()

    assert response.status_code == 503

    data = response.json()

    assert data["error"] == "http_error"
    assert (
        data["message"]
        == (
            "BOD5 prediction could not be completed because "
            "the model returned a non-finite result."
        )
    )

    assert after == before


def test_predict_returns_503_when_model_returns_infinity(
    monkeypatch,
):
    class InfiniteModel:
        def predict(self, features):
            return [math.inf]

    monkeypatch.setattr(
        main,
        "model",
        InfiniteModel(),
    )

    before = prediction_count()

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    after = prediction_count()

    assert response.status_code == 503

    data = response.json()

    assert data["error"] == "http_error"
    assert (
        data["message"]
        == (
            "BOD5 prediction could not be completed because "
            "the model returned a non-finite result."
        )
    )

    assert after == before


def test_predict_succeeds_with_working_model(
    monkeypatch,
):
    class WorkingModel:
        def predict(self, features):
            return [31.38]

    monkeypatch.setattr(
        main,
        "model",
        WorkingModel(),
    )

    before = prediction_count()

    response = client.post(
        "/predict",
        json=valid_payload(),
    )

    after = prediction_count()

    assert response.status_code == 200

    data = response.json()

    assert data["predicted_effluent_bod5"] == 31.38
    assert data["unit"] == "mg/L"

    assert after == before + 1
