import pytest

from decision.engine import generate_decision
from decision.models import DecisionInput, DecisionPriority


def make_input(**overrides):
    data = {
        "predicted_effluent_bod5": 20.0,
        "anomaly_percentile": 50.0,
        "overall_risk_level": 0,
        "overall_risk_score": 10.0,
        "dissolved_oxygen": 3.0,
        "flow_m3_day": 500.0,
        "hrt_hours": 8.0,
        "influent_bod5": 200.0,
        "influent_cod": 400.0,
        "influent_tss": 150.0,
        "model_confidence": "high",
    }

    data.update(overrides)

    return DecisionInput(**data)


def test_normal_conditions_produce_normal_priority():
    result = generate_decision(make_input())

    assert result.priority == DecisionPriority.NORMAL
    assert result.priority_name == "NORMAL"
    assert "routine" in result.summary.lower()


def test_moderate_bod5_produces_moderate_priority():
    result = generate_decision(
        make_input(predicted_effluent_bod5=40.0)
    )

    assert result.priority == DecisionPriority.MODERATE
    assert "BOD5" in " ".join(result.possible_contributors)


def test_high_bod5_produces_high_priority():
    result = generate_decision(
        make_input(predicted_effluent_bod5=60.0)
    )

    assert result.priority == DecisionPriority.HIGH
    assert "High predicted effluent BOD5" in result.possible_contributors


def test_very_high_bod5_produces_critical_priority():
    result = generate_decision(
        make_input(predicted_effluent_bod5=100.0)
    )

    assert result.priority == DecisionPriority.CRITICAL
    assert "Very high predicted effluent BOD5" in result.possible_contributors


def test_low_do_triggers_aeration_investigation():
    result = generate_decision(
        make_input(dissolved_oxygen=0.8)
    )

    assert result.priority == DecisionPriority.HIGH
    assert "Low dissolved oxygen" in result.possible_contributors
    assert any(
        "aeration" in action.lower()
        for action in result.recommended_actions
    )


def test_high_anomaly_triggers_process_investigation():
    result = generate_decision(
        make_input(anomaly_percentile=94.0)
    )

    assert result.priority == DecisionPriority.HIGH
    assert "High process anomaly score" in result.possible_contributors


def test_critical_anomaly_produces_critical_priority():
    result = generate_decision(
        make_input(anomaly_percentile=98.0)
    )

    assert result.priority == DecisionPriority.CRITICAL
    assert "Critical process anomaly score" in result.possible_contributors


def test_high_flow_triggers_hydraulic_investigation():
    result = generate_decision(
        make_input(flow_m3_day=1200.0)
    )

    assert result.priority == DecisionPriority.HIGH
    assert "Elevated hydraulic loading" in result.possible_contributors


def test_low_hrt_triggers_hydraulic_investigation():
    result = generate_decision(
        make_input(hrt_hours=1.5)
    )

    assert result.priority == DecisionPriority.HIGH
    assert "Low hydraulic retention time" in result.possible_contributors


def test_low_confidence_adds_limitation():
    result = generate_decision(
        make_input(model_confidence="research")
    )

    assert result.limitations
    assert "laboratory" in result.limitations[0].lower()


def test_multiple_triggers_are_combined():
    result = generate_decision(
        make_input(
            predicted_effluent_bod5=70.0,
            dissolved_oxygen=0.7,
            anomaly_percentile=95.0,
        )
    )

    assert result.priority == DecisionPriority.HIGH
    assert len(result.possible_contributors) >= 3
    assert result.checks_to_perform
    assert result.recommended_actions


def test_critical_risk_escalates_priority():
    result = generate_decision(
        make_input(overall_risk_level=4)
    )

    assert result.priority == DecisionPriority.CRITICAL


def test_high_risk_escalates_normal_assessment():
    result = generate_decision(
        make_input(overall_risk_level=3)
    )

    assert result.priority == DecisionPriority.HIGH


def test_duplicate_recommendations_are_removed():
    result = generate_decision(
        make_input(
            predicted_effluent_bod5=70.0,
            dissolved_oxygen=0.7,
            anomaly_percentile=95.0,
            overall_risk_level=3,
        )
    )

    assert len(result.checks_to_perform) == len(
        set(result.checks_to_perform)
    )

    assert len(result.recommended_actions) == len(
        set(result.recommended_actions)
    )


def test_invalid_nan_input_is_rejected():
    result = make_input(
        predicted_effluent_bod5=float("nan")
    )

    with pytest.raises(ValueError):
        generate_decision(result)


def test_invalid_infinite_input_is_rejected():
    result = make_input(
        dissolved_oxygen=float("inf")
    )

    with pytest.raises(ValueError):
        generate_decision(result)


def test_invalid_anomaly_percentile_is_rejected():
    result = make_input(
        anomaly_percentile=101.0
    )

    with pytest.raises(ValueError):
        generate_decision(result)


def test_invalid_flow_is_rejected():
    result = make_input(
        flow_m3_day=0.0
    )

    with pytest.raises(ValueError):
        generate_decision(result)


def test_invalid_hrt_is_rejected():
    result = make_input(
        hrt_hours=-1.0
    )

    with pytest.raises(ValueError):
        generate_decision(result)


def test_same_input_produces_same_decision():
    data = make_input(
        predicted_effluent_bod5=65.0,
        dissolved_oxygen=0.9,
        anomaly_percentile=93.0,
    )

    first = generate_decision(data)
    second = generate_decision(data)

    assert first == second


def test_engine_does_not_claim_root_cause():
    result = generate_decision(
        make_input(
            predicted_effluent_bod5=65.0,
            dissolved_oxygen=0.8,
        )
    )

    text = " ".join(
        result.possible_contributors
        + result.evidence
        + result.summary.split()
    ).lower()

    assert "caused by" not in text
    assert "proves" not in text
