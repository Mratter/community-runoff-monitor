from __future__ import annotations

from datetime import datetime, timezone

from src.clients.usgs_client import parse_usgs_instantaneous_values
from src.clients.weather_client import parse_open_meteo_snapshot


def test_usgs_parser_normalizes_mocked_instantaneous_values():
    payload = {
        "value": {
            "timeSeries": [
                {
                    "variable": {
                        "variableCode": [{"value": "00060"}],
                        "variableName": "Streamflow, ft&#179;/s",
                        "unit": {"unitCode": "ft3/s"},
                    },
                    "values": [
                        {
                            "value": [
                                {
                                    "value": "42.5",
                                    "dateTime": "2026-05-21T12:00:00.000Z",
                                    "qualifiers": ["P"],
                                }
                            ]
                        }
                    ],
                },
                {
                    "variable": {
                        "variableCode": [{"value": "63680"}],
                        "variableName": "Turbidity",
                        "unit": {"unitCode": "FNU"},
                    },
                    "values": [{"value": [{"value": "not numeric", "dateTime": "2026-05-21T12:05:00.000Z"}]}],
                },
            ]
        }
    }

    records = parse_usgs_instantaneous_values(payload)

    assert records[0]["parameter_code"] == "00060"
    assert records[0]["value"] == 42.5
    assert records[0]["qualifier"] == "P"
    assert records[1]["parameter_code"] == "63680"
    assert records[1]["value"] is None


def test_open_meteo_parser_calculates_recent_and_forecast_precipitation():
    payload = {
        "hourly": {
            "time": [
                "2026-05-21T08:00",
                "2026-05-21T09:00",
                "2026-05-21T10:00",
                "2026-05-21T11:00",
                "2026-05-21T12:00",
                "2026-05-21T13:00",
                "2026-05-21T14:00",
                "2026-05-21T15:00",
                "2026-05-21T16:00",
                "2026-05-21T17:00",
                "2026-05-21T18:00",
            ],
            "precipitation": [0, 1, 2, 3, 4, 5, 1, 1, 1, 1, 1],
            "temperature_2m": [20, 20, 20, 20, 21, 21, 21, 21, 22, 22, 22],
        }
    }
    now = datetime(2026, 5, 21, 12, tzinfo=timezone.utc)

    snapshot = parse_open_meteo_snapshot(payload, now)

    assert snapshot["source"] == "Open-Meteo"
    assert snapshot["rain_1h_mm"] == 4.0
    assert snapshot["rain_6h_mm"] == 10.0
    assert snapshot["forecast_next_6h_mm"] == 10.0
    assert snapshot["temperature_c"] == 21.0

