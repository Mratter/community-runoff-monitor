from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_active_regions, get_region_by_slug, init_db, seed_regions  # noqa: E402
from src.ml.features import build_and_store_features  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build model-ready feature rows.")
    parser.add_argument("--region", help="Region slug. Defaults to all regions.")
    args = parser.parse_args()

    conn = init_db()
    seed_regions(conn)
    regions = [get_region_by_slug(conn, args.region)] if args.region else get_active_regions(conn)
    for region in regions:
        if region is None:
            raise SystemExit(f"Unknown region slug: {args.region}")
        features, (inserted, updated) = build_and_store_features(conn, region["id"])
        print(f"{region['slug']}: {len(features)} rows, inserted={inserted}, updated={updated}")


if __name__ == "__main__":
    main()

