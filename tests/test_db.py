from __future__ import annotations

from datetime import datetime, timezone

from src.db import count_rows, init_db, seed_regions, upsert_sensor_reading


def test_duplicate_reading_upsert_updates_existing_row(tmp_path):
    db_path = tmp_path / "runoff.db"
    conn = init_db(db_path)
    seed_regions(conn)
    timestamp = datetime(2026, 5, 21, 12, tzinfo=timezone.utc)
    base = {
        "region_id": 1,
        "timestamp_utc": timestamp,
        "source": "USGS",
        "parameter_code": "00060",
        "parameter_name": "Discharge",
        "value": 10.0,
        "unit": "ft3/s",
        "qualifier": None,
        "raw_payload_hash": "first",
    }

    inserted_first = upsert_sensor_reading(conn, base)
    inserted_second = upsert_sensor_reading(conn, {**base, "value": 20.0, "raw_payload_hash": "second"})

    assert inserted_first is True
    assert inserted_second is False
    assert count_rows(conn, "sensor_readings") == 1

