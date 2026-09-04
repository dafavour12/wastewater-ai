from __future__ import annotations

from dataclasses import dataclass

import pytest

from risk.scoring import (
    RiskAssessment,
    RiskLevel,
    assess_risk,
)
from scenario.models import ScenarioInput
from scenario.risk import (
    _extract_anomaly_percentile,
    assess_scenario_risk,
)

from tests.test_scenario_models import (
    build_scenario_input,
)


def build_process_result(
    *,
    anomaly_percentile: float = 20.0,
) -> dict[str, float]:
    """
    Build a minimal process result for scenario risk tests.
    """

    return {
        "anomaly_score": 0.10,
        "anomaly_percentile": anomaly_percentile,
    }


@dataclass
class ProcessResultObject:
    """
    Minimal object-shaped process result used to verify that
    the adapter supports attribute-based process outputs.
    """

    anomaly_percentile: float


def test_extract_anomaly_percentile_from_dict():
    result = build_process_result(
        anomaly_percentile=35.0,
    )

    assert (
        _extract_anomaly_percentile(result)
        == 35.0
    )


def test_extract_anomaly_percentile_from_object():
    result = ProcessResultObject(
        anomaly_percentile=42.0,
    )

    assert (
        _extract_anomaly_percentile(result)
        == 42.0
    )


def test_extract_anomaly_percentile_requires_field():
    with pytest.raises(
        ValueError,
        match="missing.*anomaly_percentile",
    ):
        _extract_anomaly_percentile(
            {"anomaly_score": 0.1}
        )


def test_extract_anomaly_percentile_rejects_non_numeric():
    with pytest.raises(
        ValueError,
        match="invalid.*anomaly_percentile",
    ):
        _extract_anomaly_percentile(
            {"anomaly_percentile": "invalid"}
        )


@pytest.mark.parametrize(
    "percentile",
    [-0.1, 100.1],
)
def test_extract_anomaly_percentile_rejects_out_of_range(
    percentile,
):
    with pytest.raises(
        ValueError,
        match="between 0 and 100",
    ):
        _extract_anomaly_percentile(
            {
                "anomaly_percentile": percentile
            }
        )


@pytest.mark.parametrize(
    "percentile",
    [0.0, 50.0, 100.0],
)
def test_extract_anomaly_percentile_accepts_boundaries(
    percentile,
):
    assert (
        _extract_anomaly_percentile(
            {
                "anomaly_percentile": percentile
            }
        )
        == percentile
    )


def test_assess_scenario_risk_returns_risk_assessment():
    scenario = build_scenario_input()

    result = assess_scenario_risk(
        scenario,
        predicted_bod5=15.0,
        process_result=build_process_result(),
    )

    assert isinstance(
        result,
        RiskAssessment,
    )


def test_assess_scenario_risk_uses_predicted_bod5():
    scenario = build_scenario_input()

    result = assess_scenario_risk(
        scenario,
        predicted_bod5=35.0,
        process_result=build_process_result(),
    )

    assert result.predicted_bod5 == 35.0


def test_assess_scenario_risk_uses_anomaly_percentile():
    scenario = build_scenario_input()

    result = assess_scenario_risk(
        scenario,
        predicted_bod5=15.0,
        process_result=build_process_result(
            anomaly_percentile=85.0,
        ),
    )

    assert result.anomaly_percentile == 85.0


def test_assess_scenario_risk_uses_scenario_model_confidence():
    scenario = build_scenario_input(
        model_confidence="research",
    )

    result = assess_scenario_risk(
        scenario,
        predicted_bod5=15.0,
        process_result=build_process_result(),
    )

    assert result.confidence_score == 25.0


def test_assess_scenario_risk_matches_v27_engine():
    scenario = build_scenario_input(
        model_confidence="research",
    )

    predicted_bod5 = 35.0
    anomaly_percentile = 80.0

    scenario_result = assess_scenario_risk(
        scenario,
        predicted_bod5=predicted_bod5,
        process_result=build_process_result(
            anomaly_percentile=anomaly_percentile,
        ),
    )

    direct_result = assess_risk(
        predicted_bod5=predicted_bod5,
        anomaly_percentile=anomaly_percentile,
        model_confidence="research",
    )

    assert scenario_result == direct_result


def test_assess_scenario_risk_supports_object_process_result():
    scenario = build_scenario_input()

    result = assess_scenario_risk(
        scenario,
        predicted_bod5=15.0,
        process_result=ProcessResultObject(
            anomaly_percentile=25.0,
        ),
    )

    assert result.anomaly_percentile == 25.0


def test_assess_scenario_risk_rejects_invalid_predicted_bod5():
    scenario = build_scenario_input()

    with pytest.raises(
        ValueError,
        match="predicted_bod5 must be numeric",
    ):
        assess_scenario_risk(
            scenario,
            predicted_bod5="invalid",
            process_result=build_process_result(),
        )


def test_assess_scenario_risk_rejects_negative_predicted_bod5():
    scenario = build_scenario_input()

    with pytest.raises(
        ValueError,
        match="predicted_bod5 must not be negative",
    ):
        assess_scenario_risk(
            scenario,
            predicted_bod5=-1.0,
            process_result=build_process_result(),
        )


def test_assess_scenario_risk_rejects_invalid_scenario():
    with pytest.raises(
        TypeError,
        match="scenario must be a ScenarioInput",
    ):
        assess_scenario_risk(
            scenario="invalid",
            predicted_bod5=15.0,
            process_result=build_process_result(),
        )


def test_assess_scenario_risk_does_not_mutate_scenario():
    scenario = build_scenario_input()

    original_metadata = dict(
        scenario.metadata
    )
    original_wastewater = dict(
        scenario.wastewater
    )
    original_process = dict(
        scenario.process
    )

    assess_scenario_risk(
        scenario,
        predicted_bod5=15.0,
        process_result=build_process_result(
            anomaly_percentile=55.0,
        ),
    )

    assert scenario.metadata == original_metadata
    assert scenario.wastewater == original_wastewater
    assert scenario.process == original_process


def test_assess_scenario_risk_is_deterministic():
    scenario = build_scenario_input()

    process_result = build_process_result(
        anomaly_percentile=60.0,
    )

    first = assess_scenario_risk(
        scenario,
        predicted_bod5=25.0,
        process_result=process_result,
    )

    second = assess_scenario_risk(
        scenario,
        predicted_bod5=25.0,
        process_result=process_result,
    )

    assert first == second


def test_assess_scenario_risk_preserves_v27_risk_level():
    scenario = build_scenario_input()

    result = assess_scenario_risk(
        scenario,
        predicted_bod5=35.0,
        process_result=build_process_result(
            anomaly_percentile=95.0,
        ),
    )

    assert isinstance(
        result.risk_level,
        RiskLevel,
    )


def test_assess_scenario_risk_propagates_v27_errors(
    monkeypatch,
):
    scenario = build_scenario_input()

    def failing_assess_risk(
        *,
        predicted_bod5,
        anomaly_percentile,
        model_confidence,
    ):
        raise ValueError(
            "V2.7 risk engine failure"
        )

    monkeypatch.setattr(
        "scenario.risk.assess_risk",
        failing_assess_risk,
    )

    with pytest.raises(
        ValueError,
        match="V2.7 risk engine failure",
    ):
        assess_scenario_risk(
            scenario,
            predicted_bod5=15.0,
            process_result=build_process_result(),
        )


def test_scenario_risk_module_does_not_import_fastapi():
    import scenario.risk as scenario_risk

    source = open(
        scenario_risk.__file__,
        encoding="utf-8",
    ).read()

    assert "from fastapi" not in source
    assert "import fastapi" not in source
