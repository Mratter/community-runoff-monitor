from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd


def classification_metrics(y_true: Iterable[int | bool], y_score: Iterable[float], threshold: float = 0.5) -> dict[str, Any]:
    from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

    y_true_array = np.asarray(list(y_true)).astype(int)
    y_score_array = np.asarray(list(y_score), dtype=float)
    y_pred = (y_score_array >= threshold).astype(int)
    metrics = {
        "precision": float(precision_score(y_true_array, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true_array, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true_array, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true_array, y_pred, labels=[0, 1]).tolist(),
    }
    metrics["roc_auc"] = float(roc_auc_score(y_true_array, y_score_array)) if len(set(y_true_array)) > 1 else None
    return metrics


def compare_rule_and_ml(rule_category: str | None, forecast_probability: float | None, anomaly_score: float | None) -> str:
    elevated_rule = rule_category in {"Elevated", "High", "Severe"}
    elevated_ml = forecast_probability is not None and forecast_probability >= 0.5
    anomalous = anomaly_score is not None and anomaly_score >= 3.0

    if elevated_rule and elevated_ml:
        return "Rule-based score and ML model both indicate elevated runoff risk."
    if not elevated_rule and anomalous:
        return (
            "Rule-based score is low, but anomaly detection flagged an unusual stream rise. "
            "Treat this as a research signal, not an official alert."
        )
    if elevated_rule and not elevated_ml and forecast_probability is not None:
        return "Rule-based score is elevated while the ML forecast is lower; rely on the transparent rule-based score."
    if not elevated_rule and elevated_ml:
        return "ML forecast probability is elevated while the rule-based score is low; treat this as a supplementary research signal."
    return "Rule-based and ML research signals do not show a notable disagreement."


def count_positive_targets(frame: pd.DataFrame, column: str = "target_high_risk_next_6h") -> int:
    if frame.empty or column not in frame:
        return 0
    return int(frame[column].eq(True).sum())


def backtest_summary(frame: pd.DataFrame, probability_column: str = "forecast_probability") -> dict[str, Any]:
    if frame.empty or "target_high_risk_next_6h" not in frame or probability_column not in frame:
        return {"status": "insufficient_data", "metrics": None, "false_positives": [], "false_negatives": []}
    usable = frame.dropna(subset=["target_high_risk_next_6h", probability_column]).copy()
    if usable.empty:
        return {"status": "insufficient_data", "metrics": None, "false_positives": [], "false_negatives": []}
    metrics = classification_metrics(usable["target_high_risk_next_6h"].astype(int), usable[probability_column])
    usable["predicted"] = usable[probability_column] >= 0.5
    false_positives = usable[(usable["predicted"]) & (~usable["target_high_risk_next_6h"].astype(bool))].head(10)
    false_negatives = usable[(~usable["predicted"]) & (usable["target_high_risk_next_6h"].astype(bool))].head(10)
    return {
        "status": "ok",
        "metrics": metrics,
        "false_positives": false_positives.to_dict("records"),
        "false_negatives": false_negatives.to_dict("records"),
    }
