from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from html import unescape
from typing import Any

import requests

from src.config import PARAMETER_CODES


LOGGER = logging.getLogger(__name__)
USGS_IV_ENDPOINT = "https://waterservices.usgs.gov/nwis/iv/"


def _safe_float(value: Any) -> float | None:
    try:
        if value in ("", None):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def payload_hash(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def parse_usgs_instantaneous_values(payload: dict[str, Any]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    series = payload.get("value", {}).get("timeSeries", [])
    if not isinstance(series, list):
        return records

    raw_hash = payload_hash(payload)
    for item in series:
        variable = item.get("variable", {})
        code_values = variable.get("variableCode", [])
        parameter_code = None
        if code_values and isinstance(code_values, list):
            parameter_code = code_values[0].get("value")
        if not parameter_code:
            continue

        parameter_name = PARAMETER_CODES.get(parameter_code) or unescape(str(variable.get("variableName", parameter_code)))
        unit = variable.get("unit", {}).get("unitCode")
        values_groups = item.get("values", [])
        if not values_groups:
            continue
        values = values_groups[0].get("value", [])
        if not isinstance(values, list):
            continue

        for reading in values:
            timestamp = _parse_datetime(reading.get("dateTime"))
            if timestamp is None:
                continue
            qualifiers = reading.get("qualifiers") or []
            if isinstance(qualifiers, list):
                qualifier = ",".join(str(value) for value in qualifiers) if qualifiers else None
            else:
                qualifier = str(qualifiers)
            records.append(
                {
                    "timestamp_utc": timestamp,
                    "source": "USGS",
                    "parameter_code": parameter_code,
                    "parameter_name": parameter_name,
                    "value": _safe_float(reading.get("value")),
                    "unit": unit,
                    "qualifier": qualifier,
                    "raw_payload_hash": raw_hash,
                }
            )

    return records


def fetch_usgs_current_values(usgs_site_no: str, timeout: int = 15) -> list[dict[str, Any]]:
    params = {
        "format": "json",
        "sites": usgs_site_no,
        "parameterCd": ",".join(PARAMETER_CODES),
        "siteStatus": "all",
        "period": "PT6H",
    }
    try:
        response = requests.get(USGS_IV_ENDPOINT, params=params, timeout=timeout)
        response.raise_for_status()
        payload = response.json()
    except requests.RequestException as exc:
        LOGGER.warning("USGS request failed for site %s: %s", usgs_site_no, exc)
        raise
    except ValueError as exc:
        LOGGER.warning("USGS returned malformed JSON for site %s", usgs_site_no)
        raise RuntimeError("USGS returned malformed JSON") from exc

    records = parse_usgs_instantaneous_values(payload)
    LOGGER.info("USGS returned %s normalized records for site %s", len(records), usgs_site_no)
    return records

