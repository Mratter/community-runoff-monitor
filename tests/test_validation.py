from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from src.validation import ValidationError, validate_field_observation


def _valid_observation() -> dict:
    return {
        "timestamp_utc": datetime.now(timezone.utc),
        "observer_name": "Ari",
        "latitude": 38.9862,
        "longitude": -77.0049,
        "water_clarity_score": 4,
        "visual_turbidity_score": 2,
        "runoff_present": True,
        "odor_present": False,
        "trash_or_debris_present": True,
        "notes": "Outfall was flowing after rain.",
        "photo_url": "https://example.org/photo.jpg",
    }


def test_field_observation_validation_accepts_valid_payload():
    validated = validate_field_observation(_valid_observation())

    assert validated["water_clarity_score"] == 4
    assert validated["photo_url"].startswith("https://")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("latitude", 91),
        ("longitude", -181),
        ("water_clarity_score", 6),
        ("visual_turbidity_score", 0),
        ("photo_url", "ftp://example.org/photo.jpg"),
        ("notes", "x" * 1001),
    ],
)
def test_field_observation_validation_rejects_invalid_values(field, value):
    payload = _valid_observation()
    payload[field] = value

    with pytest.raises(ValidationError):
        validate_field_observation(payload)


def test_field_observation_validation_rejects_timestamp_more_than_ten_minutes_future():
    payload = _valid_observation()
    payload["timestamp_utc"] = datetime.now(timezone.utc) + timedelta(minutes=11)

    with pytest.raises(ValidationError):
        validate_field_observation(payload)

