from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


LOGGER = logging.getLogger(__name__)
MODEL_FEATURES = [
    "rain_1h_mm",
    "rain_6h_mm",
    "rain_24h_mm",
    "forecast_next_6h_mm",
    "discharge_value",
    "gage_height_value",
    "turbidity_value",
    "specific_conductance_value",
    "stream_percentile",
    "turbidity_percentile",
    "stream_rise_rate_1h",
    "stream_rise_rate_6h",
]


def score_anomalies(frame: pd.DataFrame, method: str = "rolling_zscore", min_samples: int = 100) -> pd.DataFrame:
    scored = frame.copy()
    if scored.empty:
        scored["anomaly_score"] = []
        return scored

    features = [
        column
        for column in [
            "discharge_value",
            "gage_height_value",
            "turbidity_value",
            "specific_conductance_value",
            "stream_rise_rate_1h",
        ]
        if column in scored.columns
    ]
    if not features or len(scored) < min_samples:
        scored["anomaly_score"] = np.nan
        return scored

    if method == "isolation_forest":
        try:
            from sklearn.ensemble import IsolationForest
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import make_pipeline

            model = make_pipeline(
                SimpleImputer(strategy="median"),
                IsolationForest(n_estimators=100, contamination="auto", random_state=42),
            )
            values = scored[features]
            model.fit(values)
            scored["anomaly_score"] = -model.decision_function(values)
            return scored
        except Exception as exc:  # pragma: no cover - fallback safety
            LOGGER.warning("Isolation Forest anomaly scoring failed, falling back to rolling z-score: %s", exc)

    zscores = []
    for column in features:
        series = pd.to_numeric(scored[column], errors="coerce")
        rolling_median = series.rolling(window=min(24, max(5, min_samples)), min_periods=5).median().shift(1)
        rolling_mad = (series - rolling_median).abs().rolling(window=min(24, max(5, min_samples)), min_periods=5).median().shift(1)
        fallback_std = series.rolling(window=min(24, max(5, min_samples)), min_periods=5).std().shift(1)
        denom = rolling_mad.replace(0, np.nan).fillna(fallback_std).replace(0, np.nan)
        z = ((series - rolling_median).abs() / denom).replace([np.inf, -np.inf], np.nan)
        absolute_jump = (series - rolling_median).abs()
        zscores.append(z.fillna(absolute_jump))
    scored["anomaly_score"] = pd.concat(zscores, axis=1).max(axis=1)
    return scored


def predict_with_model(model_path: str | Path, feature_row: pd.DataFrame) -> dict[str, Any] | None:
    path = Path(model_path)
    if not path.exists():
        LOGGER.warning("Model file missing: %s", path)
        return None
    try:
        model = joblib.load(path)
        X = feature_row.reindex(columns=MODEL_FEATURES)
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(X)[0][1])
        else:
            probability = float(model.predict(X)[0])
        predicted_category = "Elevated" if probability >= 0.5 else "Low"
        top_features = (
            X.iloc[0].abs().sort_values(ascending=False).dropna().head(5).round(3).to_dict()
            if not X.empty
            else {}
        )
        return {
            "forecast_probability": probability,
            "predicted_category": predicted_category,
            "top_features": top_features,
        }
    except Exception as exc:  # pragma: no cover - defensive UI/runtime path
        LOGGER.warning("Model prediction failed: %s", exc)
        return None

