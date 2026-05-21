from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from src.ml.evaluate import compare_rule_and_ml


def safe_dataframe(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    if isinstance(rows, pd.DataFrame):
        return rows
    return pd.DataFrame(rows)


def risk_badge(category: str) -> None:
    color = {
        "Low": "green",
        "Elevated": "orange",
        "High": "red",
        "Severe": "violet",
        "Insufficient Data": "gray",
    }.get(category, "gray")
    st.markdown(f":{color}-badge[{category}]")


def risk_metric_label(risk: dict[str, Any] | None) -> str:
    if not risk:
        return "Insufficient Data"
    category = risk.get("category") or "Insufficient Data"
    confidence = risk.get("confidence")
    if category == "Low" and confidence is not None and float(confidence) < 0.75:
        return "Low, limited data"
    return category


def data_quality_banner(
    latest_weather: dict[str, Any] | None,
    latest_readings: list[dict[str, Any]],
    region_note: str | None = None,
) -> None:
    issues = []
    if latest_weather is None:
        issues.append("No cached weather snapshot is available.")
    if not latest_readings:
        issues.append("No cached USGS readings are available.")
        if region_note:
            issues.append(region_note)
    latest_timestamps = []
    if latest_weather and latest_weather.get("timestamp_utc"):
        latest_timestamps.append(pd.to_datetime(latest_weather["timestamp_utc"], utc=True))
    for reading in latest_readings:
        if reading.get("timestamp_utc"):
            latest_timestamps.append(pd.to_datetime(reading["timestamp_utc"], utc=True))
    if latest_timestamps:
        newest = max(latest_timestamps)
        age_hours = (datetime.now(timezone.utc) - newest.to_pydatetime()).total_seconds() / 3600
        if age_hours > 6:
            issues.append(f"Newest cached data are about {age_hours:.1f} hours old.")
    if issues:
        st.warning(" ".join(issues))
    else:
        st.success("Data quality: recent cached stream and weather data are available.")


def component_breakdown(risk: dict[str, Any] | None) -> None:
    if not risk:
        st.info("No risk snapshot is available yet. Refresh the selected region to calculate one.")
        return
    frame = pd.DataFrame(
        [
            {"Component": "Recent rainfall", "Score": risk.get("rain_component"), "Weight": 40},
            {"Component": "Stream/gage", "Score": risk.get("stream_component"), "Weight": 30},
            {"Component": "Turbidity/proxy", "Score": risk.get("turbidity_component"), "Weight": 20},
            {"Component": "Forecast rainfall", "Score": risk.get("forecast_component"), "Weight": 10},
        ]
    )
    st.dataframe(frame, width="stretch", hide_index=True)


def risk_chart(risk_frame: pd.DataFrame, chart_key: str = "risk-chart") -> None:
    if risk_frame.empty or "score" not in risk_frame:
        st.info("No risk history is available for this range.")
        return
    chart_frame = risk_frame.dropna(subset=["score"]).copy()
    if chart_frame.empty:
        st.info("Risk history exists, but all recent scores are marked insufficient data.")
        return
    fig = px.line(chart_frame, x="timestamp_utc", y="score", color="category", markers=True, title="Rule-Based Risk Score")
    fig.update_yaxes(range=[0, 100])
    st.plotly_chart(fig, width="stretch", key=chart_key)


def sensor_chart(sensor_frame: pd.DataFrame, parameter_code: str | None = None, chart_key: str = "sensor-chart") -> None:
    if sensor_frame.empty:
        st.info("No stream sensor data are available for this range.")
        return
    frame = sensor_frame.copy()
    if parameter_code:
        frame = frame[frame["parameter_code"] == parameter_code]
    if frame.empty:
        st.info("No readings are available for the selected parameter in this range.")
        return
    fig = px.line(
        frame,
        x="timestamp_utc",
        y="value",
        color="parameter_name" if "parameter_name" in frame else "parameter_code",
        markers=True,
        title="USGS Stream Readings",
    )
    st.plotly_chart(fig, width="stretch", key=chart_key)


def rainfall_chart(weather_frame: pd.DataFrame, chart_key: str = "rainfall-chart") -> None:
    if weather_frame.empty:
        st.info("No weather/rainfall data are available for this range.")
        return
    y_columns = [column for column in ["rain_1h_mm", "rain_6h_mm", "rain_24h_mm", "forecast_next_6h_mm"] if column in weather_frame]
    if not y_columns:
        st.info("Weather snapshots do not include rainfall fields.")
        return
    fig = px.line(weather_frame, x="timestamp_utc", y=y_columns, markers=True, title="Rainfall and Forecast Precipitation")
    st.plotly_chart(fig, width="stretch", key=chart_key)


def ml_research_panel(latest_prediction: dict[str, Any] | None, latest_risk: dict[str, Any] | None) -> None:
    st.subheader("ML / Anomaly Research Signals")
    st.caption("Supplementary research signals only. These are not official warnings and do not replace the rule-based score.")
    if latest_prediction is None:
        st.info(
            "ML predictions are disabled for this region because there is not enough historical data yet. "
            "The dashboard is using the transparent rule-based score."
        )
        return

    cols = st.columns(3)
    cols[0].metric("Anomaly score", _format_optional(latest_prediction.get("anomaly_score")))
    probability = latest_prediction.get("forecast_probability")
    cols[1].metric("Forecast probability", _format_optional(probability, pct=True))
    cols[2].metric("ML category", latest_prediction.get("predicted_category") or "Unavailable")

    st.write(
        compare_rule_and_ml(
            latest_risk.get("category") if latest_risk else None,
            probability,
            latest_prediction.get("anomaly_score"),
        )
    )
    top_features = latest_prediction.get("top_features_json")
    if top_features:
        try:
            st.json(json.loads(top_features))
        except json.JSONDecodeError:
            st.write(top_features)


def _format_optional(value: Any, pct: bool = False) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    number = float(value)
    return f"{number:.0%}" if pct else f"{number:.2f}"
