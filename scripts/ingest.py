from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_active_regions, get_region_by_slug, init_db, seed_regions  # noqa: E402
from src.refresh import refresh_all_regions, refresh_region  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh public water and weather data.")
    parser.add_argument("--region", help="Region slug to refresh. Defaults to all regions.")
    args = parser.parse_args()

    conn = init_db()
    seed_regions(conn)
    if args.region:
        region = get_region_by_slug(conn, args.region)
        if region is None:
            available = ", ".join(region["slug"] for region in get_active_regions(conn))
            raise SystemExit(f"Unknown region slug '{args.region}'. Available: {available}")
        result = refresh_region(region["id"])
        print(result)
    else:
        for result in refresh_all_regions():
            print(result)


if __name__ == "__main__":
    main()

