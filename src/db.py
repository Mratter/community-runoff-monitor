from __future__ import annotations

import json
import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from src.config import DB_PATH, DEFAULT_REGIONS, ensure_data_dirs
from src.validation import validate_field_observation, validate_region_payload


LOGGER = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        dt = value
    else:
        dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _row_to_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return dict(row)


def init_db(path: str | Path = DB_PATH) -> sqlite3.Connection:
    ensure_data_dirs()
    db_path = Path(path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    return conn


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS regions (
    id INTEGER PRIMARY KEY,
    slug TEXT NOT NULL UNIQUE,
    display_name TEXT NOT NULL,
    waterway_name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    description TEXT,
    latitude REAL NOT NULL CHECK(latitude BETWEEN -90 AND 90),
    longitude REAL NOT NULL CHECK(longitude BETWEEN -180 AND 180),
    map_zoom INTEGER DEFAULT 12,
    usgs_site_no TEXT NOT NULL UNIQUE,
    timezone TEXT NOT NULL,
    is_active INTEGER DEFAULT 1,
    sort_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sensor_readings (
    id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    timestamp_utc TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'USGS',
    parameter_code TEXT NOT NULL,
    parameter_name TEXT NOT NULL,
    value REAL,
    unit TEXT,
    qualifier TEXT,
    raw_payload_hash TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(region_id, timestamp_utc, source, parameter_code)
);
CREATE INDEX IF NOT EXISTS idx_sensor_region_param_time ON sensor_readings(region_id, parameter_code, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_sensor_region_time ON sensor_readings(region_id, timestamp_utc);

CREATE TABLE IF NOT EXISTS weather_snapshots (
    id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    timestamp_utc TEXT NOT NULL,
    source TEXT NOT NULL,
    rain_1h_mm REAL CHECK(rain_1h_mm IS NULL OR rain_1h_mm >= 0),
    rain_6h_mm REAL CHECK(rain_6h_mm IS NULL OR rain_6h_mm >= 0),
    rain_24h_mm REAL CHECK(rain_24h_mm IS NULL OR rain_24h_mm >= 0),
    forecast_next_6h_mm REAL CHECK(forecast_next_6h_mm IS NULL OR forecast_next_6h_mm >= 0),
    temperature_c REAL,
    raw_payload_hash TEXT,
    created_at TEXT NOT NULL,
    UNIQUE(region_id, timestamp_utc, source)
);
CREATE INDEX IF NOT EXISTS idx_weather_region_time ON weather_snapshots(region_id, timestamp_utc);

CREATE TABLE IF NOT EXISTS risk_snapshots (
    id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    timestamp_utc TEXT NOT NULL,
    score REAL CHECK(score IS NULL OR (score >= 0 AND score <= 100)),
    category TEXT NOT NULL CHECK(category IN ('Low', 'Elevated', 'High', 'Severe', 'Insufficient Data')),
    confidence REAL NOT NULL CHECK(confidence >= 0 AND confidence <= 1),
    rain_component REAL,
    stream_component REAL,
    turbidity_component REAL,
    forecast_component REAL,
    available_weight REAL NOT NULL CHECK(available_weight >= 0 AND available_weight <= 100),
    explanation TEXT NOT NULL,
    missing_components_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_risk_region_time ON risk_snapshots(region_id, timestamp_utc);

CREATE TABLE IF NOT EXISTS field_observations (
    id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    timestamp_utc TEXT NOT NULL,
    observer_name TEXT,
    latitude REAL NOT NULL CHECK(latitude BETWEEN -90 AND 90),
    longitude REAL NOT NULL CHECK(longitude BETWEEN -180 AND 180),
    water_clarity_score INTEGER CHECK(water_clarity_score IS NULL OR water_clarity_score BETWEEN 1 AND 5),
    visual_turbidity_score INTEGER CHECK(visual_turbidity_score IS NULL OR visual_turbidity_score BETWEEN 1 AND 5),
    runoff_present INTEGER NOT NULL,
    odor_present INTEGER,
    trash_or_debris_present INTEGER,
    notes TEXT CHECK(notes IS NULL OR length(notes) <= 1000),
    photo_url TEXT CHECK(photo_url IS NULL OR photo_url LIKE 'http://%' OR photo_url LIKE 'https://%'),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_observation_region_time ON field_observations(region_id, timestamp_utc);

CREATE TABLE IF NOT EXISTS refresh_logs (
    id INTEGER PRIMARY KEY,
    region_id INTEGER REFERENCES regions(id) ON DELETE SET NULL,
    started_at_utc TEXT NOT NULL,
    finished_at_utc TEXT,
    status TEXT NOT NULL CHECK(status IN ('success', 'partial_success', 'failure')),
    source TEXT NOT NULL,
    records_inserted INTEGER DEFAULT 0,
    records_updated INTEGER DEFAULT 0,
    error_message TEXT
);

CREATE TABLE IF NOT EXISTS ml_feature_snapshots (
    id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    timestamp_utc TEXT NOT NULL,
    rain_1h_mm REAL,
    rain_6h_mm REAL,
    rain_24h_mm REAL,
    forecast_next_6h_mm REAL,
    discharge_value REAL,
    gage_height_value REAL,
    turbidity_value REAL,
    specific_conductance_value REAL,
    stream_percentile REAL,
    turbidity_percentile REAL,
    stream_rise_rate_1h REAL,
    stream_rise_rate_6h REAL,
    target_high_risk_next_6h INTEGER,
    created_at TEXT NOT NULL,
    UNIQUE(region_id, timestamp_utc)
);
CREATE INDEX IF NOT EXISTS idx_ml_features_region_time ON ml_feature_snapshots(region_id, timestamp_utc);

CREATE TABLE IF NOT EXISTS ml_model_runs (
    id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    model_type TEXT NOT NULL,
    model_version TEXT NOT NULL,
    trained_at_utc TEXT,
    training_start_utc TEXT,
    training_end_utc TEXT,
    n_samples INTEGER,
    n_positive_events INTEGER,
    metrics_json TEXT,
    feature_importance_json TEXT,
    model_path TEXT,
    status TEXT,
    notes TEXT
);

CREATE TABLE IF NOT EXISTS ml_predictions (
    id INTEGER PRIMARY KEY,
    region_id INTEGER NOT NULL REFERENCES regions(id) ON DELETE CASCADE,
    model_run_id INTEGER REFERENCES ml_model_runs(id) ON DELETE SET NULL,
    timestamp_utc TEXT NOT NULL,
    anomaly_score REAL,
    forecast_probability REAL,
    predicted_category TEXT,
    top_features_json TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_region_time ON ml_predictions(region_id, timestamp_utc);
CREATE INDEX IF NOT EXISTS idx_ml_predictions_model_run ON ml_predictions(region_id, model_run_id);
"""


def seed_regions(conn: sqlite3.Connection) -> None:
    now = _iso(_utc_now())
    for sort_order, region in enumerate(DEFAULT_REGIONS):
        cleaned = validate_region_payload(region)
        conn.execute(
            """
            INSERT INTO regions (
                slug, display_name, waterway_name, city, state, description, latitude, longitude,
                map_zoom, usgs_site_no, timezone, is_active, sort_order, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
            ON CONFLICT(slug) DO UPDATE SET
                display_name=excluded.display_name,
                waterway_name=excluded.waterway_name,
                city=excluded.city,
                state=excluded.state,
                description=excluded.description,
                latitude=excluded.latitude,
                longitude=excluded.longitude,
                map_zoom=excluded.map_zoom,
                usgs_site_no=excluded.usgs_site_no,
                timezone=excluded.timezone,
                sort_order=excluded.sort_order,
                updated_at=excluded.updated_at
            """,
            (
                cleaned["slug"],
                cleaned["display_name"],
                cleaned["waterway_name"],
                cleaned["city"],
                cleaned["state"],
                cleaned["description"],
                cleaned["latitude"],
                cleaned["longitude"],
                cleaned["map_zoom"],
                cleaned["usgs_site_no"],
                cleaned["timezone"],
                sort_order,
                now,
                now,
            ),
        )
    conn.commit()
    LOGGER.info("Seeded %s predefined regions", len(DEFAULT_REGIONS))


def get_active_regions(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    rows = conn.execute("SELECT * FROM regions WHERE is_active = 1 ORDER BY sort_order, display_name").fetchall()
    return [dict(row) for row in rows]


def get_region_by_id(conn: sqlite3.Connection, region_id: int) -> dict[str, Any] | None:
    return _row_to_dict(conn.execute("SELECT * FROM regions WHERE id = ?", (region_id,)).fetchone())


def get_region_by_slug(conn: sqlite3.Connection, slug: str) -> dict[str, Any] | None:
    return _row_to_dict(conn.execute("SELECT * FROM regions WHERE slug = ?", (slug,)).fetchone())


def count_rows(conn: sqlite3.Connection, table_name: str) -> int:
    if not table_name.replace("_", "").isalnum():
        raise ValueError("invalid table name")
    return int(conn.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0])


def upsert_sensor_reading(conn: sqlite3.Connection, reading: dict[str, Any]) -> bool:
    timestamp = _iso(reading["timestamp_utc"])
    existing = conn.execute(
        """
        SELECT id FROM sensor_readings
        WHERE region_id = ? AND timestamp_utc = ? AND source = ? AND parameter_code = ?
        """,
        (reading["region_id"], timestamp, reading.get("source", "USGS"), reading["parameter_code"]),
    ).fetchone()
    if existing:
        conn.execute(
            """
            UPDATE sensor_readings
            SET parameter_name = ?, value = ?, unit = ?, qualifier = ?, raw_payload_hash = ?
            WHERE id = ?
            """,
            (
                reading["parameter_name"],
                reading.get("value"),
                reading.get("unit"),
                reading.get("qualifier"),
                reading.get("raw_payload_hash"),
                existing["id"],
            ),
        )
        conn.commit()
        return False

    conn.execute(
        """
        INSERT INTO sensor_readings (
            region_id, timestamp_utc, source, parameter_code, parameter_name, value, unit, qualifier,
            raw_payload_hash, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            reading["region_id"],
            timestamp,
            reading.get("source", "USGS"),
            reading["parameter_code"],
            reading["parameter_name"],
            reading.get("value"),
            reading.get("unit"),
            reading.get("qualifier"),
            reading.get("raw_payload_hash"),
            _iso(_utc_now()),
        ),
    )
    conn.commit()
    return True


def upsert_weather_snapshot(conn: sqlite3.Connection, snapshot: dict[str, Any]) -> bool:
    timestamp = _iso(snapshot["timestamp_utc"])
    existing = conn.execute(
        "SELECT id FROM weather_snapshots WHERE region_id = ? AND timestamp_utc = ? AND source = ?",
        (snapshot["region_id"], timestamp, snapshot["source"]),
    ).fetchone()
    fields = (
        snapshot.get("rain_1h_mm"),
        snapshot.get("rain_6h_mm"),
        snapshot.get("rain_24h_mm"),
        snapshot.get("forecast_next_6h_mm"),
        snapshot.get("temperature_c"),
        snapshot.get("raw_payload_hash"),
    )
    if existing:
        conn.execute(
            """
            UPDATE weather_snapshots
            SET rain_1h_mm = ?, rain_6h_mm = ?, rain_24h_mm = ?, forecast_next_6h_mm = ?,
                temperature_c = ?, raw_payload_hash = ?
            WHERE id = ?
            """,
            (*fields, existing["id"]),
        )
        conn.commit()
        return False

    conn.execute(
        """
        INSERT INTO weather_snapshots (
            region_id, timestamp_utc, source, rain_1h_mm, rain_6h_mm, rain_24h_mm,
            forecast_next_6h_mm, temperature_c, raw_payload_hash, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            snapshot["region_id"],
            timestamp,
            snapshot["source"],
            *fields,
            _iso(_utc_now()),
        ),
    )
    conn.commit()
    return True


def insert_risk_snapshot(conn: sqlite3.Connection, region_id: int, risk: dict[str, Any], timestamp_utc: datetime | None = None) -> int:
    timestamp = _iso(timestamp_utc or _utc_now())
    cursor = conn.execute(
        """
        INSERT INTO risk_snapshots (
            region_id, timestamp_utc, score, category, confidence, rain_component, stream_component,
            turbidity_component, forecast_component, available_weight, explanation, missing_components_json,
            created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            region_id,
            timestamp,
            risk.get("score"),
            risk["category"],
            risk["confidence"],
            risk.get("rain_component"),
            risk.get("stream_component"),
            risk.get("turbidity_component"),
            risk.get("forecast_component"),
            risk["available_weight"],
            risk["explanation"],
            risk["missing_components_json"],
            _iso(_utc_now()),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def insert_field_observation(conn: sqlite3.Connection, region_id: int, payload: dict[str, Any]) -> int:
    observation = validate_field_observation(payload)
    cursor = conn.execute(
        """
        INSERT INTO field_observations (
            region_id, timestamp_utc, observer_name, latitude, longitude, water_clarity_score,
            visual_turbidity_score, runoff_present, odor_present, trash_or_debris_present, notes,
            photo_url, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            region_id,
            _iso(observation["timestamp_utc"]),
            observation.get("observer_name"),
            observation["latitude"],
            observation["longitude"],
            observation.get("water_clarity_score"),
            observation.get("visual_turbidity_score"),
            int(observation["runoff_present"]),
            None if observation.get("odor_present") is None else int(observation["odor_present"]),
            None
            if observation.get("trash_or_debris_present") is None
            else int(observation["trash_or_debris_present"]),
            observation.get("notes"),
            observation.get("photo_url"),
            _iso(_utc_now()),
        ),
    )
    conn.commit()
    LOGGER.info("Created field observation %s for region %s", cursor.lastrowid, region_id)
    return int(cursor.lastrowid)


def get_latest_sensor_readings(conn: sqlite3.Connection, region_id: int) -> list[dict[str, Any]]:
    rows = conn.execute(
        """
        SELECT * FROM sensor_readings
        WHERE region_id = ?
        ORDER BY timestamp_utc DESC
        """,
        (region_id,),
    ).fetchall()
    seen: set[str] = set()
    latest: list[dict[str, Any]] = []
    for row in rows:
        if row["parameter_code"] in seen:
            continue
        latest.append(dict(row))
        seen.add(row["parameter_code"])
    return latest


def get_latest_weather(conn: sqlite3.Connection, region_id: int) -> dict[str, Any] | None:
    return _row_to_dict(
        conn.execute(
            "SELECT * FROM weather_snapshots WHERE region_id = ? ORDER BY timestamp_utc DESC LIMIT 1",
            (region_id,),
        ).fetchone()
    )


def get_latest_risk(conn: sqlite3.Connection, region_id: int) -> dict[str, Any] | None:
    return _row_to_dict(
        conn.execute(
            "SELECT * FROM risk_snapshots WHERE region_id = ? ORDER BY timestamp_utc DESC LIMIT 1",
            (region_id,),
        ).fetchone()
    )


def get_recent_sensor_values(
    conn: sqlite3.Connection, region_id: int, parameter_code: str, limit: int = 200
) -> list[float]:
    rows = conn.execute(
        """
        SELECT value FROM sensor_readings
        WHERE region_id = ? AND parameter_code = ? AND value IS NOT NULL
        ORDER BY timestamp_utc DESC
        LIMIT ?
        """,
        (region_id, parameter_code, limit),
    ).fetchall()
    return [float(row["value"]) for row in rows][::-1]


def get_recent_table(conn: sqlite3.Connection, table_name: str, region_id: int, hours: int | None = None) -> pd.DataFrame:
    if not table_name.replace("_", "").isalnum():
        raise ValueError("invalid table name")
    where = "WHERE region_id = ?"
    params: list[Any] = [region_id]
    if hours is not None:
        from datetime import timedelta

        cutoff = _iso(_utc_now().replace(microsecond=0) - timedelta(hours=int(hours)))
        where += " AND timestamp_utc >= ?"
        params.append(cutoff)
    query = f"SELECT * FROM {table_name} {where} ORDER BY timestamp_utc"
    return pd.read_sql_query(query, conn, params=params, parse_dates=["timestamp_utc"])


def get_sensor_dataframe(conn: sqlite3.Connection, region_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT timestamp_utc, parameter_code, value FROM sensor_readings WHERE region_id = ? ORDER BY timestamp_utc",
        conn,
        params=(region_id,),
        parse_dates=["timestamp_utc"],
    )


def get_weather_dataframe(conn: sqlite3.Connection, region_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT timestamp_utc, rain_1h_mm, rain_6h_mm, rain_24h_mm, forecast_next_6h_mm, temperature_c
        FROM weather_snapshots WHERE region_id = ? ORDER BY timestamp_utc
        """,
        conn,
        params=(region_id,),
        parse_dates=["timestamp_utc"],
    )


def get_field_observations(conn: sqlite3.Connection, region_id: int, limit: int = 200) -> list[dict[str, Any]]:
    rows = conn.execute(
        "SELECT * FROM field_observations WHERE region_id = ? ORDER BY timestamp_utc DESC LIMIT ?",
        (region_id, limit),
    ).fetchall()
    return [dict(row) for row in rows]


def create_refresh_log(conn: sqlite3.Connection, region_id: int | None, source: str) -> int:
    cursor = conn.execute(
        """
        INSERT INTO refresh_logs (region_id, started_at_utc, status, source, records_inserted, records_updated)
        VALUES (?, ?, 'failure', ?, 0, 0)
        """,
        (region_id, _iso(_utc_now()), source),
    )
    conn.commit()
    return int(cursor.lastrowid)


def finish_refresh_log(
    conn: sqlite3.Connection,
    log_id: int,
    status: str,
    records_inserted: int = 0,
    records_updated: int = 0,
    error_message: str | None = None,
) -> None:
    conn.execute(
        """
        UPDATE refresh_logs
        SET finished_at_utc = ?, status = ?, records_inserted = ?, records_updated = ?, error_message = ?
        WHERE id = ?
        """,
        (_iso(_utc_now()), status, records_inserted, records_updated, error_message, log_id),
    )
    conn.commit()


def upsert_ml_feature_rows(conn: sqlite3.Connection, region_id: int, rows: Iterable[dict[str, Any]]) -> tuple[int, int]:
    inserted = 0
    updated = 0
    for row in rows:
        timestamp = _iso(row["timestamp_utc"])
        existing = conn.execute(
            "SELECT id FROM ml_feature_snapshots WHERE region_id = ? AND timestamp_utc = ?",
            (region_id, timestamp),
        ).fetchone()
        values = (
            row.get("rain_1h_mm"),
            row.get("rain_6h_mm"),
            row.get("rain_24h_mm"),
            row.get("forecast_next_6h_mm"),
            row.get("discharge_value"),
            row.get("gage_height_value"),
            row.get("turbidity_value"),
            row.get("specific_conductance_value"),
            row.get("stream_percentile"),
            row.get("turbidity_percentile"),
            row.get("stream_rise_rate_1h"),
            row.get("stream_rise_rate_6h"),
            None if pd.isna(row.get("target_high_risk_next_6h")) else int(bool(row.get("target_high_risk_next_6h"))),
        )
        if existing:
            conn.execute(
                """
                UPDATE ml_feature_snapshots
                SET rain_1h_mm=?, rain_6h_mm=?, rain_24h_mm=?, forecast_next_6h_mm=?,
                    discharge_value=?, gage_height_value=?, turbidity_value=?, specific_conductance_value=?,
                    stream_percentile=?, turbidity_percentile=?, stream_rise_rate_1h=?, stream_rise_rate_6h=?,
                    target_high_risk_next_6h=?
                WHERE id=?
                """,
                (*values, existing["id"]),
            )
            updated += 1
        else:
            conn.execute(
                """
                INSERT INTO ml_feature_snapshots (
                    region_id, timestamp_utc, rain_1h_mm, rain_6h_mm, rain_24h_mm, forecast_next_6h_mm,
                    discharge_value, gage_height_value, turbidity_value, specific_conductance_value,
                    stream_percentile, turbidity_percentile, stream_rise_rate_1h, stream_rise_rate_6h,
                    target_high_risk_next_6h, created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (region_id, timestamp, *values, _iso(_utc_now())),
            )
            inserted += 1
    conn.commit()
    return inserted, updated


def get_ml_feature_dataframe(conn: sqlite3.Connection, region_id: int) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM ml_feature_snapshots WHERE region_id = ? ORDER BY timestamp_utc",
        conn,
        params=(region_id,),
        parse_dates=["timestamp_utc"],
    )


def insert_ml_model_run(conn: sqlite3.Connection, region_id: int, run: dict[str, Any]) -> int:
    cursor = conn.execute(
        """
        INSERT INTO ml_model_runs (
            region_id, model_type, model_version, trained_at_utc, training_start_utc, training_end_utc,
            n_samples, n_positive_events, metrics_json, feature_importance_json, model_path, status, notes
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            region_id,
            run["model_type"],
            run["model_version"],
            _iso(run.get("trained_at_utc")),
            _iso(run.get("training_start_utc")) if run.get("training_start_utc") else None,
            _iso(run.get("training_end_utc")) if run.get("training_end_utc") else None,
            run.get("n_samples"),
            run.get("n_positive_events"),
            json.dumps(run.get("metrics", {})),
            json.dumps(run.get("feature_importance", {})) if run.get("feature_importance") is not None else None,
            run.get("model_path"),
            run.get("status"),
            run.get("notes"),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def get_latest_model_run(conn: sqlite3.Connection, region_id: int) -> dict[str, Any] | None:
    return _row_to_dict(
        conn.execute(
            "SELECT * FROM ml_model_runs WHERE region_id = ? ORDER BY trained_at_utc DESC, id DESC LIMIT 1",
            (region_id,),
        ).fetchone()
    )


def insert_ml_prediction(conn: sqlite3.Connection, prediction: dict[str, Any]) -> int:
    cursor = conn.execute(
        """
        INSERT INTO ml_predictions (
            region_id, model_run_id, timestamp_utc, anomaly_score, forecast_probability,
            predicted_category, top_features_json, created_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            prediction["region_id"],
            prediction.get("model_run_id"),
            _iso(prediction["timestamp_utc"]),
            prediction.get("anomaly_score"),
            prediction.get("forecast_probability"),
            prediction.get("predicted_category"),
            json.dumps(prediction.get("top_features", {})),
            _iso(_utc_now()),
        ),
    )
    conn.commit()
    return int(cursor.lastrowid)


def get_latest_ml_prediction(conn: sqlite3.Connection, region_id: int) -> dict[str, Any] | None:
    return _row_to_dict(
        conn.execute(
            "SELECT * FROM ml_predictions WHERE region_id = ? ORDER BY timestamp_utc DESC, id DESC LIMIT 1",
            (region_id,),
        ).fetchone()
    )
