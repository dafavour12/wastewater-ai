import sys
from pathlib import Path
from unittest.mock import Mock

import pytest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )


from dashboard.app import (  # noqa: E402
    api_get,
    api_post,
    format_number,
    get_risk_assessment,
    get_risk_history,
    get_risk_statistics,
    risk_level_indicator,
    risk_level_label,
    submit_risk_assessment,
)


def test_risk_level_label_uppercases_value():
    assert (
        risk_level_label("high")
        == "HIGH"
    )


def test_risk_level_label_replaces_underscores():
    assert (
        risk_level_label("very_high")
        == "VERY HIGH"
    )


def test_risk_level_indicator_normal():
    assert (
        risk_level_indicator("normal")
        == "🟢"
    )


def test_risk_level_indicator_low():
    assert (
        risk_level_indicator("low")
        == "🔵"
    )


def test_risk_level_indicator_elevated():
    assert (
        risk_level_indicator("elevated")
        == "🟡"
    )


def test_risk_level_indicator_high():
    assert (
        risk_level_indicator("high")
        == "🟠"
    )


def test_risk_level_indicator_critical():
    assert (
        risk_level_indicator("critical")
        == "🔴"
    )


def test_unknown_risk_level_indicator():
    assert (
        risk_level_indicator("unknown")
        == "⚪"
    )


def test_format_number():
    assert (
        format_number(42.12345)
        == "42.12"
    )


def test_format_number_custom_decimals():
    assert (
        format_number(
            42.12345,
            3,
        )
        == "42.123"
    )


def test_format_number_invalid_value():
    assert (
        format_number("invalid")
        == "N/A"
    )


def test_api_get_calls_correct_endpoint(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "status": "ok"
    }

    monkeypatch.setattr(
        "dashboard.app.requests.get",
        Mock(
            return_value=response
        ),
    )

    result = api_get(
        "/risk/assessments"
    )

    assert result == {
        "status": "ok"
    }

    requests_get = (
        sys.modules[
            "dashboard.app"
        ].requests.get
    )

    requests_get.assert_called_once_with(
        "http://127.0.0.1:8000/risk/assessments",
        timeout=10,
    )

    response.raise_for_status.assert_called_once()


def test_api_post_calls_correct_endpoint(monkeypatch):
    response = Mock()
    response.json.return_value = {
        "assessment_id": 1
    }

    requests_post = Mock(
        return_value=response
    )

    monkeypatch.setattr(
        "dashboard.app.requests.post",
        requests_post,
    )

    payload = {
        "model_confidence": "research"
    }

    result = api_post(
        "/risk/assess/process",
        payload,
    )

    assert result == {
        "assessment_id": 1
    }

    requests_post.assert_called_once_with(
        "http://127.0.0.1:8000/risk/assess/process",
        json=payload,
        timeout=30,
    )

    response.raise_for_status.assert_called_once()


def test_get_risk_statistics_uses_statistics_endpoint(
    monkeypatch,
):
    expected = {
        "total_assessments": 10
    }

    monkeypatch.setattr(
        "dashboard.app.api_get",
        lambda path: expected,
    )

    result = get_risk_statistics()

    assert result == expected


def test_get_risk_history_uses_history_endpoint(
    monkeypatch,
):
    expected = [
        {
            "id": 10
        }
    ]

    monkeypatch.setattr(
        "dashboard.app.api_get",
        lambda path: expected,
    )

    result = get_risk_history()

    assert result == expected


def test_get_risk_assessment_uses_id_endpoint(
    monkeypatch,
):
    expected = {
        "id": 25
    }

    captured = {}

    def fake_api_get(path):
        captured["path"] = path
        return expected

    monkeypatch.setattr(
        "dashboard.app.api_get",
        fake_api_get,
    )

    result = get_risk_assessment(25)

    assert result == expected
    assert (
        captured["path"]
        == "/risk/assessments/25"
    )


def test_submit_risk_assessment_uses_combined_endpoint(
    monkeypatch,
):
    expected = {
        "assessment_id": 137,
        "overall_risk_level": "HIGH",
    }

    captured = {}

    def fake_api_post(path, payload):
        captured["path"] = path
        captured["payload"] = payload

        return expected

    monkeypatch.setattr(
        "dashboard.app.api_post",
        fake_api_post,
    )

    payload = {
        "wastewater": {},
        "process": {},
        "model_confidence": "research",
    }

    result = submit_risk_assessment(
        payload
    )

    assert result == expected

    assert (
        captured["path"]
        == "/risk/assess/process"
    )

    assert (
        captured["payload"]
        == payload
    )


def test_api_get_raises_http_error(
    monkeypatch,
):
    response = Mock()

    error = RuntimeError(
        "request failed"
    )

    response.raise_for_status.side_effect = error

    monkeypatch.setattr(
        "dashboard.app.requests.get",
        Mock(
            return_value=response
        ),
    )

    with pytest.raises(RuntimeError):
        api_get("/test")


def test_api_post_raises_http_error(
    monkeypatch,
):
    response = Mock()

    error = RuntimeError(
        "request failed"
    )

    response.raise_for_status.side_effect = error

    monkeypatch.setattr(
        "dashboard.app.requests.post",
        Mock(
            return_value=response
        ),
    )

    with pytest.raises(RuntimeError):
        api_post(
            "/test",
            {},
        )
