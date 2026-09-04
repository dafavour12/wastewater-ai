from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def base_payload() -> dict:
    return {
        "predicted_effluent_bod5": 15.0,
        "anomaly_percentile": 20.0,
        "overall_risk_level": 0,
        "overall_risk_score": 0.0,
        "dissolved_oxygen": 2.0,
        "flow_m3_day": 1000.0,
        "hrt_hours": 8.0,
        "influent_bod5": 250.0,
        "influent_cod": 500.0,
        "influent_tss": 200.0,
        "model_confidence": "high",
    }


def test_decision_recommend_normal_case():
    response = client.post(
        "/decision/recommend",
        json=base_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert "priority" in data
    assert "summary" in data
    assert "possible_contributors" in data
    assert "checks_to_perform" in data
    assert "recommended_actions" in data
    assert "monitoring_recommendations" in data
    assert "evidence" in data
    assert "limitations" in data


def test_decision_recommend_returns_expected_priority_for_normal_case():
    response = client.post(
        "/decision/recommend",
        json=base_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] == "normal"


def test_decision_recommend_high_bod5():
    payload = base_payload()
    payload["predicted_effluent_bod5"] = 40.0

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] in {
        "moderate",
        "high",
        "critical",
    }

    assert len(data["recommended_actions"]) > 0


def test_decision_recommend_very_high_bod5():
    payload = base_payload()
    payload["predicted_effluent_bod5"] = 100.0

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] == "critical"
    assert len(data["checks_to_perform"]) > 0
    assert len(data["recommended_actions"]) > 0


def test_decision_recommend_low_dissolved_oxygen():
    payload = base_payload()
    payload["dissolved_oxygen"] = 0.5

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] in {
        "high",
        "critical",
    }

    combined_text = " ".join(
        data["checks_to_perform"]
        + data["recommended_actions"]
        + data["possible_contributors"]
    ).lower()

    assert (
        "aeration" in combined_text
        or "oxygen" in combined_text
        or "do" in combined_text
    )


def test_decision_recommend_high_anomaly():
    payload = base_payload()
    payload["anomaly_percentile"] = 97.0

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] in {
        "high",
        "critical",
    }

    assert len(data["checks_to_perform"]) > 0


def test_decision_recommend_critical_anomaly():
    payload = base_payload()
    payload["anomaly_percentile"] = 99.9

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] == "critical"


def test_decision_recommend_high_overall_risk():
    payload = base_payload()
    payload["overall_risk_level"] = 3
    payload["overall_risk_score"] = 75.0

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] in {
        "high",
        "critical",
    }


def test_decision_recommend_critical_overall_risk():
    payload = base_payload()
    payload["overall_risk_level"] = 4
    payload["overall_risk_score"] = 95.0

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert data["priority"] == "critical"


def test_decision_recommend_low_confidence_adds_limitation():
    payload = base_payload()
    payload["model_confidence"] = "low"

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["limitations"]) > 0

    limitation_text = " ".join(
        data["limitations"]
    ).lower()

    assert (
        "confidence" in limitation_text
        or "verify" in limitation_text
        or "laboratory" in limitation_text
    )


def test_decision_recommend_limited_confidence_adds_limitation():
    payload = base_payload()
    payload["model_confidence"] = "limited"

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert len(data["limitations"]) > 0


def test_decision_recommend_invalid_anomaly_percentile():
    payload = base_payload()
    payload["anomaly_percentile"] = 101.0

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 422


def test_decision_recommend_negative_bod5():
    payload = base_payload()
    payload["predicted_effluent_bod5"] = -1.0

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 422


def test_decision_recommend_invalid_flow():
    payload = base_payload()
    payload["flow_m3_day"] = 0.0

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 422


def test_decision_recommend_invalid_hrt():
    payload = base_payload()
    payload["hrt_hours"] = 0.0

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 422


def test_decision_recommend_missing_required_field():
    payload = base_payload()

    del payload["influent_bod5"]

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 422


def test_decision_recommend_is_deterministic():
    payload = base_payload()

    first_response = client.post(
        "/decision/recommend",
        json=payload,
    )

    second_response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert first_response.status_code == 200
    assert second_response.status_code == 200

    assert first_response.json() == second_response.json()


def test_decision_recommend_does_not_require_database():
    payload = base_payload()

    response = client.post(
        "/decision/recommend",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "assessment_id" not in data
    assert "id" not in data


def test_decision_recommend_response_lists_are_lists():
    response = client.post(
        "/decision/recommend",
        json=base_payload(),
    )

    assert response.status_code == 200

    data = response.json()

    assert isinstance(
        data["possible_contributors"],
        list,
    )

    assert isinstance(
        data["checks_to_perform"],
        list,
    )

    assert isinstance(
        data["recommended_actions"],
        list,
    )

    assert isinstance(
        data["monitoring_recommendations"],
        list,
    )

    assert isinstance(
        data["evidence"],
        list,
    )

    assert isinstance(
        data["limitations"],
        list,
    )
