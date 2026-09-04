import pytest

from risk.scoring import (
    RiskLevel,
    assess_risk,
    calculate_anomaly_score,
    calculate_confidence_score,
    calculate_prediction_score,
    level_from_anomaly,
    level_from_prediction,
)


def test_prediction_score_normal():
    assert calculate_prediction_score(20) == 0.0


def test_prediction_score_low():
    assert calculate_prediction_score(25) == 25.0


def test_prediction_score_elevated():
    assert calculate_prediction_score(40) == 50.0


def test_prediction_score_high():
    assert calculate_prediction_score(60) == 75.0


def test_prediction_score_critical():
    assert calculate_prediction_score(100) > 75.0


def test_prediction_score_is_capped_at_100():
    assert calculate_prediction_score(1000) == 100.0


def test_prediction_score_rejects_negative_values():
    with pytest.raises(ValueError):
        calculate_prediction_score(-1)


def test_anomaly_score_matches_percentile():
    assert calculate_anomaly_score(98.5) == 98.5


def test_anomaly_score_rejects_values_below_zero():
    with pytest.raises(ValueError):
        calculate_anomaly_score(-0.1)


def test_anomaly_score_rejects_values_above_100():
    with pytest.raises(ValueError):
        calculate_anomaly_score(100.1)


def test_confidence_score_research_is_conservative():
    assert calculate_confidence_score("research") == 25.0


def test_confidence_score_limited_is_conservative():
    assert calculate_confidence_score("limited") == 25.0


def test_confidence_score_high():
    assert calculate_confidence_score("high") == 100.0


def test_prediction_level_normal():
    assert level_from_prediction(20) == RiskLevel.NORMAL


def test_prediction_level_low():
    assert level_from_prediction(21) == RiskLevel.LOW


def test_prediction_level_elevated():
    assert level_from_prediction(31) == RiskLevel.ELEVATED


def test_prediction_level_high():
    assert level_from_prediction(51) == RiskLevel.HIGH


def test_prediction_level_critical():
    assert level_from_prediction(76) == RiskLevel.CRITICAL


def test_anomaly_level_normal():
    assert level_from_anomaly(50) == RiskLevel.NORMAL


def test_anomaly_level_low():
    assert level_from_anomaly(90) == RiskLevel.LOW


def test_anomaly_level_elevated():
    assert level_from_anomaly(97) == RiskLevel.ELEVATED


def test_anomaly_level_high():
    assert level_from_anomaly(99) == RiskLevel.HIGH


def test_anomaly_level_critical():
    assert level_from_anomaly(99.5) == RiskLevel.CRITICAL


def test_normal_case():
    assessment = assess_risk(
        predicted_bod5=18,
        anomaly_percentile=50,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.NORMAL
    assert assessment.risk_score < 30


def test_low_case():
    assessment = assess_risk(
        predicted_bod5=25,
        anomaly_percentile=50,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.LOW


def test_elevated_prediction():
    assessment = assess_risk(
        predicted_bod5=40,
        anomaly_percentile=50,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.ELEVATED


def test_high_prediction():
    assessment = assess_risk(
        predicted_bod5=60,
        anomaly_percentile=50,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.HIGH


def test_critical_prediction():
    assessment = assess_risk(
        predicted_bod5=100,
        anomaly_percentile=50,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.CRITICAL


def test_high_anomaly_can_drive_risk():
    assessment = assess_risk(
        predicted_bod5=20,
        anomaly_percentile=99.2,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.HIGH


def test_critical_anomaly_overrides_low_prediction():
    assessment = assess_risk(
        predicted_bod5=20,
        anomaly_percentile=99.7,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.CRITICAL


def test_combined_high_anomaly_and_elevated_prediction_is_critical():
    assessment = assess_risk(
        predicted_bod5=40,
        anomaly_percentile=99.1,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.CRITICAL


def test_high_anomaly_with_normal_prediction_is_not_critical():
    assessment = assess_risk(
        predicted_bod5=20,
        anomaly_percentile=99.1,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.HIGH


def test_reason_is_generated():
    assessment = assess_risk(
        predicted_bod5=40,
        anomaly_percentile=98,
        model_confidence="research",
    )

    assert assessment.risk_reason
    assert len(assessment.risk_reason) > 10


def test_recommended_action_is_generated():
    assessment = assess_risk(
        predicted_bod5=40,
        anomaly_percentile=98,
        model_confidence="research",
    )

    assert assessment.recommended_action
    assert len(assessment.recommended_action) > 10


def test_assessment_contains_component_scores():
    assessment = assess_risk(
        predicted_bod5=35,
        anomaly_percentile=98,
        model_confidence="research",
    )

    assert 0 <= assessment.risk_score <= 100
    assert 0 <= assessment.prediction_score <= 100
    assert 0 <= assessment.anomaly_score <= 100
    assert 0 <= assessment.confidence_score <= 100


def test_case_insensitive_confidence():
    assessment = assess_risk(
        predicted_bod5=20,
        anomaly_percentile=20,
        model_confidence="RESEARCH",
    )

    assert assessment.confidence_score == 25.0


def test_invalid_confidence_is_rejected():
    with pytest.raises(ValueError):
        assess_risk(
            predicted_bod5=20,
            anomaly_percentile=20,
            model_confidence="unknown",
        )


def test_invalid_predicted_bod5_is_rejected():
    with pytest.raises(ValueError):
        assess_risk(
            predicted_bod5=-1,
            anomaly_percentile=20,
            model_confidence="research",
        )


def test_invalid_anomaly_percentile_is_rejected():
    with pytest.raises(ValueError):
        assess_risk(
            predicted_bod5=20,
            anomaly_percentile=101,
            model_confidence="research",
        )


def test_critical_anomaly_forces_high_numeric_score():
    assessment = assess_risk(
        predicted_bod5=10,
        anomaly_percentile=99.7,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.CRITICAL
    assert assessment.risk_score >= 90


def test_combined_critical_rule_forces_high_numeric_score():
    assessment = assess_risk(
        predicted_bod5=40,
        anomaly_percentile=99.1,
        model_confidence="research",
    )

    assert assessment.risk_level == RiskLevel.CRITICAL
    assert assessment.risk_score >= 85
    