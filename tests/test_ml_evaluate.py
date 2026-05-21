from __future__ import annotations

import warnings

import pandas as pd

from src.ml.evaluate import classification_metrics, compare_rule_and_ml, count_positive_targets


def test_evaluation_metrics_include_confusion_matrix_and_f1():
    metrics = classification_metrics([0, 0, 1, 1], [0.1, 0.3, 0.8, 0.9])

    assert metrics["precision"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_rule_based_vs_ml_comparison_describes_agreement_and_disagreement():
    agreement = compare_rule_and_ml("Elevated", 0.8, anomaly_score=None)
    disagreement = compare_rule_and_ml("Low", 0.2, anomaly_score=4.0)

    assert "both indicate elevated" in agreement.lower()
    assert "research signal" in disagreement.lower()


def test_count_positive_targets_handles_missing_values_without_future_warning():
    frame = pd.DataFrame({"target_high_risk_next_6h": pd.Series([True, False, None, 1, 0], dtype=object)})

    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        positives = count_positive_targets(frame)

    assert positives == 2
