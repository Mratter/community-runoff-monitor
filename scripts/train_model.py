from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_ml_feature_dataframe, get_region_by_slug, init_db, insert_ml_model_run, seed_regions  # noqa: E402
from src.ml.features import build_and_store_features  # noqa: E402
from src.ml.train import train_forecast_model  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a supplementary runoff-risk forecast model.")
    parser.add_argument("--region", default="sligo-creek-md", help="Region slug to train.")
    parser.add_argument("--model-type", default="logistic_regression", choices=["logistic_regression", "random_forest"])
    args = parser.parse_args()

    conn = init_db()
    seed_regions(conn)
    region = get_region_by_slug(conn, args.region)
    if region is None:
        raise SystemExit(f"Unknown region slug: {args.region}")

    build_and_store_features(conn, region["id"])
    features = get_ml_feature_dataframe(conn, region["id"])
    result = train_forecast_model(features, region["slug"], model_type=args.model_type)
    if result["status"] == "success":
        run_id = insert_ml_model_run(conn, region["id"], result)
        print(f"Trained model run {run_id}: {result['metrics']}")
    else:
        print(result["notes"])


if __name__ == "__main__":
    main()

