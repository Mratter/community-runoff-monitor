from __future__ import annotations

import json
from statistics import mean
from typing import Any


WEIGHTS = {
    "rainfall": 40.0,
    "stream": 30.0,
    "turbidity": 20.0,
    "forecast": 10.0,
}


def classify_score(score: float | None) -> str:
    if score is None:
        return "Insufficient Data"
    if score < 25:
        return "Low"
    if score < 50:
        return "Elevated"
    if score < 75:
        return "High"
    return "Severe"


def percentile_rank(value: float, baseline: list[float]) -> float | None:
    clean = [float(item) for item in baseline if item is not None]
    if not clean:
        return None
    return 100.0 * sum(1 for item in clean if item <= value) / len(clean)


def _rain_component(weather: dict[str, Any]) -> float | None:
    rain_values = [weather.get("rain_1h_mm"), weather.get("rain_6h_mm"), weather.get("rain_24h_mm")]
    if all(value is None for value in rain_values):
        return None
    rain_1h = max(float(weather.get("rain_1h_mm") or 0.0), 0.0)
    rain_6h = max(float(weather.get("rain_6h_mm") or 0.0), 0.0)
    rain_24h = max(float(weather.get("rain_24h_mm") or 0.0), 0.0)
    intensity = 0.5 * (rain_1h / 12.7) + 0.3 * (rain_6h / 25.4) + 0.2 * (rain_24h / 50.8)
    return round(WEIGHTS["rainfall"] * min(1.0, intensity), 3)


def _stream_component(readings: dict[str, Any], baselines: dict[str, list[float]]) -> tuple[float | None, bool, str | None]:
    parameter_code = "00060" if readings.get("00060") is not None else "00065"
    value = readings.get(parameter_code)
    baseline = baselines.get(parameter_code, [])
    if value is None or not baseline:
        return None, False, parameter_code
    percentile = percentile_rank(float(value), baseline)
    if percentile is None:
        return None, False, parameter_code
    if percentile < 60:
        score = 0.0
    elif percentile < 80:
        score = 10.0
    elif percentile < 90:
        score = 20.0
    else:
        score = 30.0
    return score, len(baseline) < 20, parameter_code


def _turbidity_component(readings: dict[str, Any], baselines: dict[str, list[float]]) -> tuple[float | None, bool]:
    value = readings.get("63680")
    baseline = baselines.get("63680", [])
    if value is None or not baseline:
        return None, False
    percentile = percentile_rank(float(value), baseline)
    if percentile is None:
        return None, False
    if percentile < 75:
        score = 0.0
    elif percentile < 90:
        score = 10.0
    else:
        score = 20.0
    return score, len(baseline) < 20


def _forecast_component(weather: dict[str, Any]) -> float | None:
    value = weather.get("forecast_next_6h_mm")
    if value is None:
        return None
    return round(WEIGHTS["forecast"] * min(1.0, max(float(value), 0.0) / 25.4), 3)


def _explain(
    category: str,
    components: dict[str, float | None],
    missing: list[str],
    weak_baselines: list[str],
    score: float | None,
    weather_only: bool = False,
) -> str:
    if category == "Insufficient Data":
        if weather_only:
            return (
                "Risk is marked Insufficient Data because this is a weather-only snapshot. "
                "Rainfall and forecast data can describe storm potential, but current stream/gage "
                "and turbidity readings are missing. "
                f"Missing components: {', '.join(missing) if missing else 'unknown'}."
            )
        return (
            "Risk could not be calculated reliably because key data are missing. "
            f"Missing components: {', '.join(missing) if missing else 'unknown'}."
        )

    reasons: list[str] = []
    rain = components.get("rainfall")
    stream = components.get("stream")
    turbidity = components.get("turbidity")
    forecast = components.get("forecast")
    if rain is not None:
        reasons.append("recent rainfall is high" if rain >= 20 else "recent rainfall is limited")
    if stream is not None:
        reasons.append("stream levels are above the recent baseline" if stream >= 20 else "stream levels are near normal")
    if turbidity is not None:
        reasons.append("turbidity is elevated" if turbidity >= 10 else "turbidity is near its recent baseline")
    if forecast is not None and forecast > 0:
        reasons.append("additional rainfall is forecast soon")

    sentence = f"Risk is {category.lower()} because " + " and ".join(reasons[:3]) + "."
    if not reasons:
        sentence = f"Risk is {category.lower()} based on the available inputs."
    if missing:
        sentence += f" Missing components: {', '.join(missing)}."
    if weak_baselines:
        sentence += f" Confidence is reduced because the {', '.join(weak_baselines)} weak baseline has limited history."
    if score is not None and category in {"High", "Severe"}:
        sentence += " Treat this as a community awareness signal, not an official warning."
    return sentence


def calculate_risk(
    weather: dict[str, Any] | None,
    readings: dict[str, Any] | None,
    baselines: dict[str, list[float]] | None,
) -> dict[str, Any]:
    weather = weather or {}
    readings = readings or {}
    baselines = baselines or {}

    components: dict[str, float | None] = {
        "rainfall": _rain_component(weather),
        "stream": None,
        "turbidity": None,
        "forecast": _forecast_component(weather),
    }
    stream_score, weak_stream, _ = _stream_component(readings, baselines)
    turbidity_score, weak_turbidity = _turbidity_component(readings, baselines)
    components["stream"] = stream_score
    components["turbidity"] = turbidity_score

    missing = [name for name, score in components.items() if score is None]
    available_weight = round(sum(WEIGHTS[name] for name, score in components.items() if score is not None), 3)
    weak_baselines = []
    if weak_stream and components["stream"] is not None:
        weak_baselines.append("stream/gage")
    if weak_turbidity and components["turbidity"] is not None:
        weak_baselines.append("turbidity")

    weather_only = (
        available_weight > 0
        and components["rainfall"] is not None
        and components["forecast"] is not None
        and components["stream"] is None
        and components["turbidity"] is None
    )

    if available_weight < 50 or weather_only:
        score = None
        category = "Insufficient Data"
    else:
        component_sum = sum(score for score in components.values() if score is not None)
        score = round(component_sum / available_weight * 100.0, 1)
        category = classify_score(score)

    confidence = available_weight / 100.0
    confidence = max(0.0, confidence - 0.1 * len(weak_baselines))
    confidence = round(min(confidence, 1.0), 3)
    explanation = _explain(category, components, missing, weak_baselines, score, weather_only=weather_only)
    baseline_values = [value for values in baselines.values() for value in values]

    return {
        "score": score,
        "category": category,
        "confidence": confidence,
        "rain_component": components["rainfall"],
        "stream_component": components["stream"],
        "turbidity_component": components["turbidity"],
        "forecast_component": components["forecast"],
        "available_weight": available_weight,
        "explanation": explanation,
        "missing_components": missing,
        "missing_components_json": json.dumps(missing),
        "baseline_mean": mean(baseline_values) if baseline_values else None,
    }
