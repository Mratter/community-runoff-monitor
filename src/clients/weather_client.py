from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import requests

from src.validation import validate_weather_snapshot


LOGGER = logging.getLogger(__name__)
OPEN_METEO_ENDPOINT = "https://api.open-meteo.com/v1/forecast"


def _payload_hash(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def _parse_hour(value: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _sum_between(rows: list[tuple[datetime, float | None]], start: datetime, end: datetime, include_start: bool) -> float | None:
    values = []
    for timestamp, precip in rows:
        if precip is None:
            continue
        after_start = timestamp >= start if include_start else timestamp > start
        if after_start and timestamp <= end:
            values.append(max(float(precip), 0.0))
    if not values:
        return None
    return round(sum(values), 3)


def parse_open_meteo_snapshot(payload: dict[str, Any], now_utc: datetime | None = None) -> dict[str, Any]:
    now = (now_utc or datetime.now(timezone.utc)).astimezone(timezone.utc).replace(minute=0, second=0, microsecond=0)
    hourly = payload.get("hourly", {})
    times = hourly.get("time", []) or []
    precipitation = hourly.get("precipitation") or hourly.get("rain") or []
    temperatures = hourly.get("temperature_2m") or []

    rows: list[tuple[datetime, float | None]] = []
    temp_by_time: dict[datetime, float] = {}
    for index, raw_time in enumerate(times):
        timestamp = _parse_hour(str(raw_time))
        if timestamp is None:
            continue
        precip_value = None
        if index < len(precipitation) and precipitation[index] is not None:
            try:
                precip_value = max(float(precipitation[index]), 0.0)
            except (TypeError, ValueError):
                precip_value = None
        rows.append((timestamp, precip_value))
        if index < len(temperatures) and temperatures[index] is not None:
            try:
                temp_by_time[timestamp] = float(temperatures[index])
            except (TypeError, ValueError):
                pass

    snapshot = {
        "timestamp_utc": now,
        "source": "Open-Meteo",
        "rain_1h_mm": _sum_between(rows, now - timedelta(hours=1), now, include_start=False),
        "rain_6h_mm": _sum_between(rows, now - timedelta(hours=6), now, include_start=False),
        "rain_24h_mm": _sum_between(rows, now - timedelta(hours=24), now, include_start=False),
        "forecast_next_6h_mm": _sum_between(rows, now, now + timedelta(hours=6), include_start=False),
        "temperature_c": temp_by_time.get(now),
        "raw_payload_hash": _payload_hash(payload),
    }
    return validate_weather_snapshot(snapshot)


def fetch_weather_snapshot(latitude: float, longitude: float, timezone_name: str, timeout: int = 15) -> dict[str, Any]:
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "hourly": "temperature_2m,precipitation",
        "past_days": 2,
        "forecast_days": 2,
        "timezone": "UTC",
    }
    try:
        response = requests.get(OPEN_METEO_ENDPOINT, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        LOGGER.warning("Open-Meteo request failed for %s,%s: %s", latitude, longitude, exc)
        raise
    except ValueError as exc:
        LOGGER.warning("Open-Meteo returned malformed JSON")
        raise RuntimeError("Open-Meteo returned malformed JSON") from exc

    snapshot = parse_open_meteo_snapshot(payload)
    snapshot["source"] = "Open-Meteo"
    LOGGER.info("Open-Meteo snapshot fetched for %s,%s (%s)", latitude, longitude, timezone_name)
    return snapshot

