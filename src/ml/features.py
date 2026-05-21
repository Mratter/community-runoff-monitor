from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from src import db
from src.ml.evaluate import count_positive_targets


LOGGER = logging.getLogger(__name__)
PARAMETER_TO_FEATURE = {
    "00060": "discharge_value",
    "00065": "gage_height_value",
    "63680": "turbidity_value",
    "00095": "specific_conductance_value",
}
FEATURE_COLUMNS = [
    "timestamp_utc",
    "rain_1h_mm",
    "rain_6h_mm",
    "rain_24h_mm",
    "forecast_next_6h_mm",
    "discharge_value",
    "gage_height_value",
    "turbidity_value",
    "specific_conductance_value",
    "stream_percentile",
    "turbidity_percentile",
    "stream_rise_rate_1h",
    "stream_rise_rate_6h",
    "target_high_risk_next_6h",
]


def _empty_features() -> pd.DataFrame:
    return pd.DataFrame(columns=FEATURE_COLUMNS)


def _percentile_series(series: pd.Series) -> pd.Series:
    if series.dropna().empty:
        return pd.Series(np.nan, index=series.index)
    return series.rank(pct=True) * 100.0


def _future_crosses_threshold(series: pd.Series, threshold: float, horizon_rows: int = 6) -> pd.Series:
    targets: list[Any] = []
    for index in range(len(series)):
        future = series.iloc[index + 1 : index + 1 + horizon_rows].dropna()
        if future.empty or pd.isna(series.iloc[index]):
            targets.append(np.nan)
        else:
            targets.append(bool((future >= threshold).any()))
    return pd.Series(targets, index=series.index)


def build_feature_frame(
    sensor_readings: pd.DataFrame,
    weather_snapshots: pd.DataFrame,
    threshold_percentile: float = 0.90,
) -> pd.DataFrame:
    if sensor_readings.empty and weather_snapshots.empty:
        return _empty_features()

    sensor = sensor_readings.copy()
    if not sensor.empty:
        sensor["timestamp_utc"] = pd.to_datetime(sensor["timestamp_utc"], utc=True)
        sensor["value"] = pd.to_numeric(sensor["value"], errors="coerce")
        sensor = sensor[sensor["parameter_code"].isin(PARAMETER_TO_FEATURE)]
        pivot = (
            sensor.pivot_table(index="timestamp_utc", columns="parameter_code", values="value", aggfunc="last")
            .rename(columns=PARAMETER_TO_FEATURE)
            .reset_index()
        )
    else:
        pivot = pd.DataFrame(columns=["timestamp_utc", *PARAMETER_TO_FEATURE.values()])

    weather = weather_snapshots.copy()
    if not weather.empty:
        weather["timestamp_utc"] = pd.to_datetime(weather["timestamp_utc"], utc=True)
        keep = ["timestamp_utc", "rain_1h_mm", "rain_6h_mm", "rain_24h_mm", "forecast_next_6h_mm"]
        weather = weather[[column for column in keep if column in weather.columns]]
    else:
        weather = pd.DataFrame(columns=["timestamp_utc", "rain_1h_mm", "rain_6h_mm", "rain_24h_mm", "forecast_next_6h_mm"])

    if pivot.empty:
        frame = weather
    elif weather.empty:
        frame = pivot
    else:
        frame = pd.merge(pivot, weather, on="timestamp_utc", how="outer")

    for column in FEATURE_COLUMNS:
        if column not in frame.columns:
            frame[column] = np.nan

    frame = frame.sort_values("timestamp_utc").reset_index(drop=True)
    stream = frame["discharge_value"].combine_first(frame["gage_height_value"])
    frame["stream_percentile"] = _percentile_series(stream)
    frame["turbidity_percentile"] = _percentile_series(frame["turbidity_value"])
    frame["stream_rise_rate_1h"] = stream.diff(1)
    frame["stream_rise_rate_6h"] = stream.diff(6)

    if stream.dropna().empty:
        frame["target_high_risk_next_6h"] = np.nan
    else:
        threshold = stream.quantile(threshold_percentile)
        frame["target_high_risk_next_6h"] = _future_crosses_threshold(stream, threshold)

    numeric_columns = [column for column in FEATURE_COLUMNS if column != "timestamp_utc"]
    for column in numeric_columns:
        if column != "target_high_risk_next_6h":
            frame[column] = pd.to_numeric(frame[column], errors="coerce")

    return frame[FEATURE_COLUMNS]


def has_enough_ml_data(frame: pd.DataFrame, task: str = "anomaly") -> bool:
    if task == "anomaly":
        return len(frame.dropna(how="all")) >= 100
    if task == "forecast":
        if len(frame) < 500 or "target_high_risk_next_6h" not in frame:
            return False
        positives = count_positive_targets(frame)
        return positives >= 20
    if task == "evaluation":
        positives = count_positive_targets(frame)
        return positives >= 20
    raise ValueError(f"unknown ML task: {task}")


def build_and_store_features(conn, region_id: int) -> tuple[pd.DataFrame, tuple[int, int]]:
    sensor = db.get_sensor_dataframe(conn, region_id)
    weather = db.get_weather_dataframe(conn, region_id)
    features = build_feature_frame(sensor, weather)
    if features.empty:
        LOGGER.info("No feature rows available for region %s", region_id)
        return features, (0, 0)
    inserted_updated = db.upsert_ml_feature_rows(conn, region_id, features.to_dict("records"))
    LOGGER.info(
        "Stored ML feature rows for region %s: inserted=%s updated=%s",
        region_id,
        inserted_updated[0],
        inserted_updated[1],
    )
    return features, inserted_updated
