from __future__ import annotations

from src.risk import calculate_risk, classify_score


def test_risk_score_with_complete_data_is_severe_when_all_components_high():
    result = calculate_risk(
        weather={
            "rain_1h_mm": 12.7,
            "rain_6h_mm": 25.4,
            "rain_24h_mm": 50.8,
            "forecast_next_6h_mm": 12.7,
        },
        readings={"00060": 100.0, "63680": 50.0},
        baselines={"00060": list(range(1, 101)), "63680": list(range(1, 51))},
    )

    assert result["category"] == "Severe"
    assert result["score"] == 95.0
    assert result["confidence"] == 1.0
    assert result["rain_component"] == 40.0
    assert result["stream_component"] == 30.0
    assert "recent rainfall" in result["explanation"].lower()


def test_risk_score_with_missing_rainfall_still_calculates_when_enough_other_data_exists():
    result = calculate_risk(
        weather={"forecast_next_6h_mm": 0.0},
        readings={"00060": 10.0, "63680": 5.0},
        baselines={"00060": list(range(1, 101)), "63680": list(range(1, 101))},
    )

    assert result["category"] == "Low"
    assert result["score"] == 0.0
    assert result["available_weight"] == 60.0
    assert "rainfall" in result["missing_components"]


def test_risk_score_with_missing_turbidity_records_missing_component():
    result = calculate_risk(
        weather={
            "rain_1h_mm": 0.0,
            "rain_6h_mm": 0.0,
            "rain_24h_mm": 0.0,
            "forecast_next_6h_mm": 0.0,
        },
        readings={"00065": 1.0},
        baselines={"00065": list(range(1, 101))},
    )

    assert result["category"] == "Low"
    assert result["available_weight"] == 80.0
    assert "turbidity" in result["missing_components"]


def test_insufficient_data_category_when_available_weight_is_too_low():
    result = calculate_risk(weather={}, readings={}, baselines={})

    assert result["category"] == "Insufficient Data"
    assert result["score"] is None
    assert result["confidence"] == 0.0


def test_weather_only_inputs_are_marked_insufficient_data():
    result = calculate_risk(
        weather={
            "rain_1h_mm": 0.0,
            "rain_6h_mm": 0.0,
            "rain_24h_mm": 0.0,
            "forecast_next_6h_mm": 0.0,
        },
        readings={},
        baselines={"00060": [], "00065": [], "63680": []},
    )

    assert result["category"] == "Insufficient Data"
    assert result["score"] is None
    assert result["available_weight"] == 50.0
    assert result["baseline_mean"] is None
    assert result["missing_components"] == ["stream", "turbidity"]
    assert "weather-only" in result["explanation"].lower()


def test_category_thresholds():
    assert classify_score(0) == "Low"
    assert classify_score(24) == "Low"
    assert classify_score(25) == "Elevated"
    assert classify_score(50) == "High"
    assert classify_score(75) == "Severe"


def test_weak_baseline_reduces_confidence_and_explanation_mentions_it():
    result = calculate_risk(
        weather={
            "rain_1h_mm": 0.0,
            "rain_6h_mm": 0.0,
            "rain_24h_mm": 0.0,
            "forecast_next_6h_mm": 0.0,
        },
        readings={"00060": 6.0, "63680": 6.0},
        baselines={"00060": [1, 2, 3, 4, 5], "63680": [1, 2, 3, 4, 5]},
    )

    assert result["available_weight"] == 100.0
    assert 0 < result["confidence"] < 1.0
    assert "weak baseline" in result["explanation"].lower()
