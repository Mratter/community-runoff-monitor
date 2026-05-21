from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.db import get_latest_model_run, get_ml_feature_dataframe, get_region_by_slug, init_db, seed_regions  # noqa: E402
from src.ml.evaluate import backtest_summary  # noqa: E402
from src.ml.predict import MODEL_FEATURES  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the latest supplementary ML model.")
    parser.add_argument("--region", default="sligo-creek-md", help="Region slug to evaluate.")
    args = parser.parse_args()

    conn = init_db()
    seed_regions(conn)
    region = get_region_by_slug(conn, args.region)
    if region is None:
        raise SystemExit(f"Unknown region slug: {args.region}")

    model_run = get_latest_model_run(conn, region["id"])
    if not model_run or not model_run.get("model_path"):
        raise SystemExit("No trained model is available for this region yet.")
    model_path = Path(model_run["model_path"])
    if not model_path.exists():
        raise SystemExit(f"Model file is missing: {model_path}")

    features = get_ml_feature_dataframe(conn, region["id"])
    if features.empty:
        raise SystemExit("No ML feature rows are available for this region.")
    model = joblib.load(model_path)
    probabilities = model.predict_proba(features.reindex(columns=MODEL_FEATURES))[:, 1]
    scored = features.copy()
    scored["forecast_probability"] = pd.Series(probabilities, index=scored.index)
    summary = backtest_summary(scored)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()

