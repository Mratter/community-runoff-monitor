from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.config import MODELS_DIR
from src.ml.evaluate import classification_metrics, count_positive_targets
from src.ml.features import has_enough_ml_data
from src.ml.predict import MODEL_FEATURES


LOGGER = logging.getLogger(__name__)


def _feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        return {name: float(value) for name, value in zip(feature_names, model.feature_importances_)}
    estimator = getattr(model, "named_steps", {}).get("logisticregression") if hasattr(model, "named_steps") else None
    if estimator is not None and hasattr(estimator, "coef_"):
        return {name: float(value) for name, value in zip(feature_names, estimator.coef_[0])}
    return {}


def train_forecast_model(
    frame: pd.DataFrame,
    region_slug: str,
    model_type: str = "logistic_regression",
    output_dir: Path = MODELS_DIR,
) -> dict[str, Any]:
    if not has_enough_ml_data(frame, task="forecast"):
        return {
            "status": "insufficient_data",
            "notes": (
                "ML predictions are disabled for this region because there is not enough historical data yet. "
                "The dashboard is using the transparent rule-based score."
            ),
            "n_samples": int(len(frame)),
            "n_positive_events": count_positive_targets(frame),
        }

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    usable = frame.dropna(subset=["target_high_risk_next_6h"]).copy()
    X = usable.reindex(columns=MODEL_FEATURES)
    y = usable["target_high_risk_next_6h"].astype(bool).astype(int)

    stratify = y if y.nunique() > 1 else None
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=stratify)
    if model_type == "random_forest":
        model = make_pipeline(SimpleImputer(strategy="median"), RandomForestClassifier(n_estimators=200, random_state=42))
    else:
        model = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_test)[:, 1]
    metrics = classification_metrics(y_test, probabilities)

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    model_path = output_dir / f"{region_slug}-{model_type}-{timestamp}.joblib"
    joblib.dump(model, model_path)

    return {
        "status": "success",
        "model_type": model_type,
        "model_version": timestamp,
        "trained_at_utc": datetime.now(timezone.utc),
        "training_start_utc": usable["timestamp_utc"].min() if "timestamp_utc" in usable else None,
        "training_end_utc": usable["timestamp_utc"].max() if "timestamp_utc" in usable else None,
        "n_samples": int(len(usable)),
        "n_positive_events": int(y.sum()),
        "metrics": metrics,
        "feature_importance": _feature_importance(model[-1] if hasattr(model, "__getitem__") else model, MODEL_FEATURES),
        "model_path": str(model_path),
        "notes": "Supplementary research model. Public dashboard should continue to prioritize rule-based scoring.",
    }
