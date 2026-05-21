from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.ml.features import build_feature_frame, has_enough_ml_data
from src.ml.predict import predict_with_model, score_anomalies


def test_ml_feature_generation_aligns_weather_and_sensor_rows():
    timestamps = pd.date_range(datetime(2026, 5, 21, tzinfo=timezone.utc), periods=12, freq="h")
    sensor_rows = []
    for i, ts in enumerate(timestamps):
        sensor_rows.append({"timestamp_utc": ts, "parameter_code": "00060", "value": float(i + 1)})
        sensor_rows.append({"timestamp_utc": ts, "parameter_code": "63680", "value": float(i + 2)})
    weather = pd.DataFrame(
        {
            "timestamp_utc": timestamps,
            "rain_1h_mm": range(12),
            "rain_6h_mm": range(12),
            "rain_24h_mm": range(12),
            "forecast_next_6h_mm": range(12),
        }
    )

    features = build_feature_frame(pd.DataFrame(sensor_rows), weather)

    assert "discharge_value" in features.columns
    assert "stream_rise_rate_1h" in features.columns
    assert "target_high_risk_next_6h" in features.columns
    assert features["target_high_risk_next_6h"].notna().any()


def test_ml_insufficient_data_rules():
    frame = pd.DataFrame({"target_high_risk_next_6h": [False] * 99})

    assert has_enough_ml_data(frame, task="anomaly") is False
    assert has_enough_ml_data(pd.DataFrame({"target_high_risk_next_6h": [False] * 100}), task="anomaly") is True
    assert has_enough_ml_data(frame, task="forecast") is False


def test_anomaly_scoring_flags_unusual_spike():
    frame = pd.DataFrame(
        {
            "discharge_value": [10.0] * 30 + [200.0],
            "gage_height_value": [1.0] * 31,
            "turbidity_value": [5.0] * 31,
            "specific_conductance_value": [100.0] * 31,
            "stream_rise_rate_1h": [0.0] * 30 + [190.0],
        }
    )

    scored = score_anomalies(frame, method="rolling_zscore", min_samples=10)

    assert scored["anomaly_score"].iloc[-1] > scored["anomaly_score"].median()


def test_model_prediction_missing_file_fallback(tmp_path):
    prediction = predict_with_model(tmp_path / "missing.joblib", pd.DataFrame([{}]))

    assert prediction is None
