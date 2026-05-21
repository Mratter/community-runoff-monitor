from __future__ import annotations

from datetime import datetime, timezone

from src.config import DEFAULT_REGION_SLUG, DEFAULT_REGIONS, get_region_data_note
from src.db import (
    get_active_regions,
    get_latest_sensor_readings,
    init_db,
    seed_regions,
    upsert_sensor_reading,
)


def test_region_seeding_creates_all_default_regions(tmp_path):
    db_path = tmp_path / "runoff.db"
    conn = init_db(db_path)

    seed_regions(conn)
    regions = get_active_regions(conn)

    assert [region["slug"] for region in regions] == [region["slug"] for region in DEFAULT_REGIONS]
    assert regions[0]["slug"] == DEFAULT_REGION_SLUG
    assert all(region["usgs_site_no"] for region in regions)


def test_rock_creek_documents_current_usgs_instantaneous_value_limitation():
    note = get_region_data_note("rock-creek-dc")

    assert note is not None
    assert "instantaneous" in note.lower()


def test_selected_region_queries_filter_sensor_readings(tmp_path):
    db_path = tmp_path / "runoff.db"
    conn = init_db(db_path)
    seed_regions(conn)
    regions = get_active_regions(conn)
    sligo = regions[0]
    rock = regions[1]
    timestamp = datetime(2026, 5, 21, 12, tzinfo=timezone.utc)

    upsert_sensor_reading(
        conn,
        {
            "region_id": sligo["id"],
            "timestamp_utc": timestamp,
            "source": "USGS",
            "parameter_code": "00060",
            "parameter_name": "Discharge",
            "value": 12.0,
            "unit": "ft3/s",
            "qualifier": None,
            "raw_payload_hash": "a",
        },
    )
    upsert_sensor_reading(
        conn,
        {
            "region_id": rock["id"],
            "timestamp_utc": timestamp,
            "source": "USGS",
            "parameter_code": "00060",
            "parameter_name": "Discharge",
            "value": 99.0,
            "unit": "ft3/s",
            "qualifier": None,
            "raw_payload_hash": "b",
        },
    )

    sligo_readings = get_latest_sensor_readings(conn, sligo["id"])

    assert len(sligo_readings) == 1
    assert sligo_readings[0]["value"] == 12.0
