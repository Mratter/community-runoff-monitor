from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


class ValidationError(ValueError):
    """Raised when user-provided or API-normalized data fails validation."""


def parse_datetime_utc(value: Any) -> datetime:
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, str):
        normalized = value.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(normalized)
        except ValueError as exc:
            raise ValidationError("timestamp must be an ISO datetime") from exc
    else:
        raise ValidationError("timestamp is required")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def validate_latitude(value: Any) -> float:
    try:
        latitude = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("latitude must be numeric") from exc
    if not -90 <= latitude <= 90:
        raise ValidationError("latitude must be between -90 and 90")
    return latitude


def validate_longitude(value: Any) -> float:
    try:
        longitude = float(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError("longitude must be numeric") from exc
    if not -180 <= longitude <= 180:
        raise ValidationError("longitude must be between -180 and 180")
    return longitude


def validate_optional_score(value: Any, field_name: str) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValidationError(f"{field_name} must be an integer from 1 to 5")
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"{field_name} must be an integer from 1 to 5") from exc
    if score < 1 or score > 5:
        raise ValidationError(f"{field_name} must be an integer from 1 to 5")
    return score


def validate_region_payload(payload: dict[str, Any]) -> dict[str, Any]:
    required = ["slug", "display_name", "waterway_name", "city", "state", "usgs_site_no", "timezone"]
    cleaned = dict(payload)
    for field in required:
        if not str(cleaned.get(field, "")).strip():
            raise ValidationError(f"{field} is required")
        cleaned[field] = str(cleaned[field]).strip()
    cleaned["latitude"] = validate_latitude(cleaned.get("latitude"))
    cleaned["longitude"] = validate_longitude(cleaned.get("longitude"))
    cleaned["map_zoom"] = int(cleaned.get("map_zoom", 12))
    cleaned["description"] = str(cleaned.get("description", "")).strip()
    return cleaned


def validate_weather_snapshot(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    for field in ["rain_1h_mm", "rain_6h_mm", "rain_24h_mm", "forecast_next_6h_mm"]:
        value = cleaned.get(field)
        if value is not None:
            value = float(value)
            if value < 0:
                raise ValidationError(f"{field} cannot be negative")
            cleaned[field] = value
    return cleaned


def validate_field_observation(payload: dict[str, Any]) -> dict[str, Any]:
    cleaned = dict(payload)
    timestamp_utc = parse_datetime_utc(cleaned.get("timestamp_utc"))
    if timestamp_utc > datetime.now(timezone.utc) + timedelta(minutes=10):
        raise ValidationError("timestamp cannot be more than 10 minutes in the future")

    cleaned["timestamp_utc"] = timestamp_utc
    cleaned["latitude"] = validate_latitude(cleaned.get("latitude"))
    cleaned["longitude"] = validate_longitude(cleaned.get("longitude"))
    cleaned["water_clarity_score"] = validate_optional_score(cleaned.get("water_clarity_score"), "water_clarity_score")
    cleaned["visual_turbidity_score"] = validate_optional_score(
        cleaned.get("visual_turbidity_score"), "visual_turbidity_score"
    )
    cleaned["runoff_present"] = bool(cleaned.get("runoff_present"))

    for field in ["odor_present", "trash_or_debris_present"]:
        if cleaned.get(field) in ("", None):
            cleaned[field] = None
        else:
            cleaned[field] = bool(cleaned[field])

    notes = cleaned.get("notes")
    cleaned["notes"] = None if notes in ("", None) else str(notes).strip()
    if cleaned["notes"] is not None and len(cleaned["notes"]) > 1000:
        raise ValidationError("notes must be 1000 characters or fewer")

    photo_url = cleaned.get("photo_url")
    cleaned["photo_url"] = None if photo_url in ("", None) else str(photo_url).strip()
    if cleaned["photo_url"] and not cleaned["photo_url"].startswith(("http://", "https://")):
        raise ValidationError("photo_url must start with http:// or https://")

    observer_name = cleaned.get("observer_name")
    cleaned["observer_name"] = None if observer_name in ("", None) else str(observer_name).strip()
    return cleaned

