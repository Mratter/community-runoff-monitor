from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from src.clients.usgs_client import fetch_usgs_current_values
from src.clients.weather_client import fetch_weather_snapshot
from src.config import DB_PATH
from src.db import (
    create_refresh_log,
    finish_refresh_log,
    get_active_regions,
    get_latest_model_run,
    get_latest_sensor_readings,
    get_latest_weather,
    get_recent_sensor_values,
    get_region_by_id,
    init_db,
    insert_ml_prediction,
    insert_risk_snapshot,
    seed_regions,
    upsert_sensor_reading,
    upsert_weather_snapshot,
)
from src.ml.features import build_and_store_features
from src.ml.predict import predict_with_model, score_anomalies
from src.risk import calculate_risk


LOGGER = logging.getLogger(__name__)


def _current_readings_by_code(latest_readings: list[dict[str, Any]]) -> dict[str, float | None]:
    return {reading["parameter_code"]: reading.get("value") for reading in latest_readings}


def _baselines_by_code(conn, region_id: int, parameter_codes: list[str]) -> dict[str, list[float]]:
    return {code: get_recent_sensor_values(conn, region_id, code, limit=200) for code in parameter_codes}


def calculate_and_store_current_risk(conn, region_id: int) -> dict[str, Any]:
    latest_readings = get_latest_sensor_readings(conn, region_id)
    latest_weather = get_latest_weather(conn, region_id) or {}
    readings = _current_readings_by_code(latest_readings)
    baselines = _baselines_by_code(conn, region_id, ["00060", "00065", "63680"])
    risk = calculate_risk(latest_weather, readings, baselines)
    insert_risk_snapshot(conn, region_id, risk, datetime.now(timezone.utc))
    LOGGER.info("Stored risk snapshot for region %s: %s", region_id, risk["category"])
    return risk


def _maybe_store_ml_prediction(conn, region_id: int, features: pd.DataFrame) -> dict[str, Any] | None:
    if features.empty:
        return None
    latest_feature = features.tail(1).copy()
    anomaly_score = None
    try:
        anomaly_frame = score_anomalies(features, method="rolling_zscore", min_samples=100)
        latest_anomaly = anomaly_frame["anomaly_score"].iloc[-1]
        anomaly_score = None if pd.isna(latest_anomaly) else float(latest_anomaly)
    except Exception as exc:  # pragma: no cover - defensive runtime path
        LOGGER.warning("ML anomaly scoring failed for region %s: %s", region_id, exc)

    latest_model = get_latest_model_run(conn, region_id)
    model_prediction = None
    if latest_model and latest_model.get("model_path"):
        model_prediction = predict_with_model(latest_model["model_path"], latest_feature)

    if anomaly_score is None and model_prediction is None:
        return None

    prediction = {
        "region_id": region_id,
        "model_run_id": latest_model["id"] if latest_model else None,
        "timestamp_utc": latest_feature["timestamp_utc"].iloc[0],
        "anomaly_score": anomaly_score,
        "forecast_probability": model_prediction.get("forecast_probability") if model_prediction else None,
        "predicted_category": model_prediction.get("predicted_category") if model_prediction else None,
        "top_features": model_prediction.get("top_features", {}) if model_prediction else {},
    }
    insert_ml_prediction(conn, prediction)
    LOGGER.info("Stored ML prediction row for region %s", region_id)
    return prediction


def refresh_region(region_id: int, db_path=DB_PATH) -> dict[str, Any]:
    conn = init_db(db_path)
    seed_regions(conn)
    region = get_region_by_id(conn, region_id)
    if region is None:
        raise ValueError(f"Unknown region_id: {region_id}")

    log_id = create_refresh_log(conn, region_id, source="USGS/Open-Meteo")
    records_inserted = 0
    records_updated = 0
    errors: list[str] = []

    try:
        try:
            records = fetch_usgs_current_values(region["usgs_site_no"])
            for record in records:
                inserted = upsert_sensor_reading(conn, {**record, "region_id": region_id})
                records_inserted += int(inserted)
                records_updated += int(not inserted)
        except Exception as exc:
            LOGGER.warning("USGS refresh failed for region %s: %s", region_id, exc)
            errors.append(f"USGS: {exc}")

        try:
            snapshot = fetch_weather_snapshot(region["latitude"], region["longitude"], region["timezone"])
            inserted = upsert_weather_snapshot(conn, {**snapshot, "region_id": region_id})
            records_inserted += int(inserted)
            records_updated += int(not inserted)
        except Exception as exc:
            LOGGER.warning("Weather refresh failed for region %s: %s", region_id, exc)
            errors.append(f"Weather: {exc}")

        risk = calculate_and_store_current_risk(conn, region_id)
        features, (feature_inserted, feature_updated) = build_and_store_features(conn, region_id)
        records_inserted += feature_inserted
        records_updated += feature_updated
        ml_prediction = _maybe_store_ml_prediction(conn, region_id, features)

        status = "partial_success" if errors else "success"
        finish_refresh_log(conn, log_id, status, records_inserted, records_updated, "; ".join(errors) or None)
        return {
            "status": status,
            "records_inserted": records_inserted,
            "records_updated": records_updated,
            "errors": errors,
            "risk": risk,
            "ml_prediction": ml_prediction,
        }
    except Exception as exc:
        LOGGER.exception("Refresh failed for region %s", region_id)
        finish_refresh_log(conn, log_id, "failure", records_inserted, records_updated, str(exc))
        return {
            "status": "failure",
            "records_inserted": records_inserted,
            "records_updated": records_updated,
            "errors": errors + [str(exc)],
            "risk": None,
            "ml_prediction": None,
        }


def refresh_all_regions(db_path=DB_PATH) -> list[dict[str, Any]]:
    conn = init_db(db_path)
    seed_regions(conn)
    results = []
    for region in get_active_regions(conn):
        results.append({"region": region["slug"], **refresh_region(region["id"], db_path=db_path)})
    return results

