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


def test_region_catalog_replaces_weak_rock_creek_with_waimanalo_stream():
    slugs = [region["slug"] for region in DEFAULT_REGIONS]

    assert "rock-creek-dc" not in slugs
    assert "waimanalo-stream-hi" in slugs


def test_waimanalo_documents_turbidity_richness():
    note = get_region_data_note("waimanalo-stream-hi")

    assert note is not None
    assert "turbidity" in note.lower()


def test_region_seeding_deactivates_retired_rock_creek_from_existing_database(tmp_path):
    db_path = tmp_path / "runoff.db"
    conn = init_db(db_path)
    timestamp = datetime(2026, 5, 21, 12, tzinfo=timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO regions (
            slug, display_name, waterway_name, city, state, description, latitude, longitude,
            map_zoom, usgs_site_no, timezone, is_active, sort_order, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
        """,
        (
            "rock-creek-dc",
            "Rock Creek — Washington, DC",
            "Rock Creek",
            "Washington",
            "DC",
            "Retired weak live-data region.",
            38.9725,
            -77.0400,
            13,
            "01648000",
            "America/New_York",
            1,
            timestamp,
            timestamp,
        ),
    )
    conn.commit()

    seed_regions(conn)
    active_slugs = [region["slug"] for region in get_active_regions(conn)]

    assert "rock-creek-dc" not in active_slugs
    assert "waimanalo-stream-hi" in active_slugs


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
