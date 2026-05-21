from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DB_PATH = DATA_DIR / "runoff_monitor.db"

load_dotenv(BASE_DIR / ".env")

APP_NAME = "Community Water Quality and Runoff Monitor"
DEFAULT_REGION_SLUG = "sligo-creek-md"
ADMIN_TOKEN_ENV = "ADMIN_TOKEN"

PARAMETER_CODES = {
    "00060": "Discharge",
    "00065": "Gage height",
    "00010": "Water temperature",
    "00095": "Specific conductance",
    "00300": "Dissolved oxygen",
    "00400": "pH",
    "63680": "Turbidity",
}

DEFAULT_REGIONS = [
    {
        "slug": "sligo-creek-md",
        "display_name": "Sligo Creek — Takoma Park, MD",
        "waterway_name": "Sligo Creek",
        "city": "Takoma Park / Silver Spring",
        "state": "MD",
        "usgs_site_no": "01650800",
        "latitude": 38.9862,
        "longitude": -77.0049,
        "timezone": "America/New_York",
        "map_zoom": 13,
        "description": (
            "Small urban creek in the Washington, DC region; good default site for a local runoff "
            "and water-quality dashboard."
        ),
    },
    {
        "slug": "rock-creek-dc",
        "display_name": "Rock Creek — Washington, DC",
        "waterway_name": "Rock Creek",
        "city": "Washington",
        "state": "DC",
        "usgs_site_no": "01648000",
        "latitude": 38.9725,
        "longitude": -77.0400,
        "timezone": "America/New_York",
        "map_zoom": 13,
        "description": "Urban creek in Rock Creek Park with a long USGS streamflow record.",
    },
    {
        "slug": "peachtree-creek-ga",
        "display_name": "Peachtree Creek — Atlanta, GA",
        "waterway_name": "Peachtree Creek",
        "city": "Atlanta",
        "state": "GA",
        "usgs_site_no": "02336300",
        "latitude": 33.8170,
        "longitude": -84.4070,
        "timezone": "America/New_York",
        "map_zoom": 12,
        "description": "Urban Atlanta watershed with strong runoff and flooding relevance.",
    },
    {
        "slug": "brays-bayou-tx",
        "display_name": "Brays Bayou — Houston, TX",
        "waterway_name": "Brays Bayou",
        "city": "Houston",
        "state": "TX",
        "usgs_site_no": "08075000",
        "latitude": 29.6960,
        "longitude": -95.4120,
        "timezone": "America/Chicago",
        "map_zoom": 12,
        "description": "Urban Houston bayou with strong stormwater and flood-monitoring relevance.",
    },
    {
        "slug": "la-river-sepulveda-ca",
        "display_name": "Los Angeles River — Sepulveda Dam, CA",
        "waterway_name": "Los Angeles River",
        "city": "Los Angeles / Van Nuys",
        "state": "CA",
        "usgs_site_no": "11092450",
        "latitude": 34.1670,
        "longitude": -118.4750,
        "timezone": "America/Los_Angeles",
        "map_zoom": 12,
        "description": (
            "Continuous water-quality monitoring site with strong turbidity, sediment, and runoff relevance."
        ),
    },
]

REGION_DATA_NOTES = {
    "sligo-creek-md": "Current USGS instantaneous values are usually available for discharge and gage context.",
    "rock-creek-dc": (
        "USGS site 01648000 has a long historical streamflow record, but the legacy USGS instantaneous-values "
        "endpoint currently returns no current readings for the parameters this app requests."
    ),
    "peachtree-creek-ga": "Current USGS instantaneous values are usually available for stream and water-quality context.",
    "brays-bayou-tx": "Current USGS instantaneous values are usually available for stream/gage context.",
    "la-river-sepulveda-ca": "Current USGS instantaneous values are usually available for water-quality context.",
}


def get_admin_token() -> str | None:
    token = os.getenv(ADMIN_TOKEN_ENV)
    return token if token else None


def get_region_data_note(slug: str | None) -> str | None:
    if not slug:
        return None
    return REGION_DATA_NOTES.get(slug)


def ensure_data_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
