from pathlib import Path

import pytest

from process.monitoring import (
    MODEL_PATH,
    PROCESS_FEATURES,
    ProcessMonitor,
    classify_alert_level,
    classify_risk_band,
    validate_process_input,
)


VALID_PROCESS_DATA = {
    "PH-P": 7.0,
    "DBO-P": 25.0,
    "SS-P": 30.0,
    "SSV-P": 20.0,
    "SED-P": 5.0,
    "COND-P": 500.0,
    "PH-D": 7.2,
    "DBO-D": 20.0,
    "DQO-D": 50.0,
    "SS-D": 25.0,
    "SSV-D": 18.0,
    "SED-D": 4.0,
    "COND-D": 480.0,
    "RD-DBO-P": 1.0,
    "RD-SS-P": 1.0,
    "RD-DBO-D": 1.0,
    "RD-SS-D": 1.0,
    "RD-DBO-G": 1.0,
    "RD-SS-G": 1.0,
    "RD-SED-G": 1.0,
    "RD-N-NH4": 1.0,
    "RD-N-NO2": 1.0,
}


def test_process_feature_count():
    assert len(PROCESS_FEATURES) == 22


def test_process_model_exists():
    assert Path(MODEL_PATH).exists()


def test_validate_process_input_accepts_valid_data():
    validate_process_input(VALID_PROCESS_DATA)


def test_validate_process_input_rejects_missing_feature():
    data = VALID_PROCESS_DATA.copy()
    data.pop("PH-P")

    with pytest.raises(ValueError, match="Missing process variables"):
        validate_process_input(data)


def test_validate_process_input_rejects_nan():
    data = VALID_PROCESS_DATA.copy()
    data["PH-P"] = float("nan")

    with pytest.raises(ValueError, match="Invalid process values"):
        validate_process_input(data)


def test_validate_process_input_rejects_infinity():
    data = VALID_PROCESS_DATA.copy()
    data["PH-P"] = float("inf")

    with pytest.raises(ValueError, match="Invalid process values"):
        validate_process_input(data)


def test_classify_risk_band_normal():
    assert classify_risk_band(50) == "normal"


def test_classify_risk_band_low():
    assert classify_risk_band(90) == "low"


def test_classify_risk_band_elevated():
    assert classify_risk_band(97) == "elevated"


def test_classify_risk_band_high():
    assert classify_risk_band(99) == "high"


def test_classify_risk_band_critical():
    assert classify_risk_band(99.5) == "critical"


def test_classify_alert_level_normal():
    assert classify_alert_level(90) == "normal"


def test_classify_alert_level_watch():
    assert classify_alert_level(97) == "watch"


def test_classify_alert_level_alert():
    assert classify_alert_level(99) == "alert"


def test_process_monitor_returns_result():
    monitor = ProcessMonitor()

    result = monitor.predict(VALID_PROCESS_DATA)

    assert 0 <= result.anomaly_percentile <= 100
    assert isinstance(result.anomaly_score, float)
    assert isinstance(result.is_anomaly, bool)
    assert result.risk_band in {
        "normal",
        "low",
        "elevated",
        "high",
        "critical",
    }
    assert result.alert_level in {
        "normal",
        "watch",
        "alert",
    }
    assert isinstance(result.message, str)
    assert result.message


def test_process_monitor_result_is_deterministic():
    monitor = ProcessMonitor()

    result_1 = monitor.predict(VALID_PROCESS_DATA)
    result_2 = monitor.predict(VALID_PROCESS_DATA)

    assert result_1.anomaly_score == result_2.anomaly_score
    assert result_1.anomaly_percentile == result_2.anomaly_percentile
    assert result_1.is_anomaly == result_2.is_anomaly
    assert result_1.risk_band == result_2.risk_band
    assert result_1.alert_level == result_2.alert_level


def test_process_monitor_rejects_missing_input():
    monitor = ProcessMonitor()

    data = VALID_PROCESS_DATA.copy()
    data.pop("DBO-P")

    with pytest.raises(ValueError, match="Missing process variables"):
        monitor.predict(data)


def test_process_monitor_rejects_invalid_input():
    monitor = ProcessMonitor()

    data = VALID_PROCESS_DATA.copy()
    data["SS-P"] = float("nan")

    with pytest.raises(ValueError, match="Invalid process values"):
        monitor.predict(data)