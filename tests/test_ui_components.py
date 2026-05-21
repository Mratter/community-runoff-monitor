from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.ui import components


def test_risk_chart_passes_explicit_streamlit_key(monkeypatch):
    captured = {}

    def fake_plotly_chart(fig, **kwargs):
        captured["key"] = kwargs.get("key")

    monkeypatch.setattr(components.st, "plotly_chart", fake_plotly_chart)
    frame = pd.DataFrame(
        [
            {
                "timestamp_utc": datetime(2026, 5, 21, tzinfo=timezone.utc),
                "score": 10.0,
                "category": "Low",
            }
        ]
    )

    components.risk_chart(frame, chart_key="dashboard-risk-chart")

    assert captured["key"] == "dashboard-risk-chart"


def test_limited_low_risk_label_mentions_limited_data():
    label = components.risk_metric_label({"category": "Low", "confidence": 0.5})

    assert label == "Low, limited data"


def test_streamlit_calls_do_not_use_deprecated_container_width_argument():
    project_root = Path(components.__file__).parents[2]
    paths = [
        project_root / "app.py",
        project_root / "src" / "ui" / "components.py",
    ]

    for path in paths:
        assert "use_container_width" not in path.read_text(encoding="utf-8")
