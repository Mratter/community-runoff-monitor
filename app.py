from __future__ import annotations

import json
import logging
from datetime import datetime, time, timezone
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.express as px
import pydeck as pdk
import streamlit as st

from src.config import APP_NAME, DEFAULT_REGION_SLUG, get_admin_token, get_region_data_note
from src.db import (
    count_rows,
    get_active_regions,
    get_field_observations,
    get_latest_ml_prediction,
    get_latest_model_run,
    get_latest_risk,
    get_latest_sensor_readings,
    get_latest_weather,
    get_ml_feature_dataframe,
    get_recent_table,
    get_region_by_slug,
    init_db,
    insert_field_observation,
    seed_regions,
)
from src.ml.evaluate import count_positive_targets
from src.refresh import refresh_all_regions, refresh_region
from src.ui.components import (
    component_breakdown,
    data_quality_banner,
    ml_research_panel,
    rainfall_chart,
    risk_badge,
    risk_chart,
    risk_metric_label,
    sensor_chart,
)
from src.validation import ValidationError


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
LOGGER = logging.getLogger(__name__)


@st.cache_resource
def get_connection():
    conn = init_db()
    seed_regions(conn)
    return conn


def _selected_region(conn, regions: list[dict]) -> dict:
    default_region = get_region_by_slug(conn, DEFAULT_REGION_SLUG) or regions[0]
    default_index = next((index for index, region in enumerate(regions) if region["id"] == default_region["id"]), 0)
    labels = [region["display_name"] for region in regions]
    if "selected_region_id" in st.session_state:
        current_index = next(
            (index for index, region in enumerate(regions) if region["id"] == st.session_state["selected_region_id"]),
            default_index,
        )
    else:
        current_index = default_index
    selected_label = st.sidebar.selectbox("Select monitoring region:", labels, index=current_index)
    selected = regions[labels.index(selected_label)]
    st.session_state["selected_region_id"] = selected["id"]
    return selected


def _admin_controls(region: dict) -> tuple[bool, str | None]:
    configured_token = get_admin_token()
    admin_token = st.sidebar.text_input("Admin token:", type="password")
    is_admin = bool(configured_token and admin_token and admin_token == configured_token)
    if configured_token is None:
        st.sidebar.warning("ADMIN_TOKEN is not set. Admin actions are disabled. Add it to .env or deployment settings.")
        st.sidebar.caption("Local preload: run `python scripts/ingest.py` to fetch public data for all predefined regions.")
    elif admin_token and not is_admin:
        st.sidebar.error("Admin token did not match.")

    if st.sidebar.button("Refresh Selected Region", disabled=not is_admin):
        with st.spinner(f"Refreshing {region['display_name']}..."):
            result = refresh_region(region["id"])
        if result["status"] == "failure":
            st.sidebar.error("Refresh failed. Cached data are still shown where available.")
        elif result["status"] == "partial_success":
            st.sidebar.warning("Refresh partially succeeded. Some cached data may be stale.")
        else:
            st.sidebar.success("Refresh complete.")
        st.sidebar.caption(f"Inserted {result['records_inserted']}; updated {result['records_updated']}.")

    if st.sidebar.button("Refresh All Regions", disabled=not is_admin):
        with st.spinner("Refreshing all configured regions..."):
            results = refresh_all_regions()
        failures = [result for result in results if result["status"] == "failure"]
        if failures:
            st.sidebar.warning(f"Completed with {len(failures)} failure(s).")
        else:
            st.sidebar.success("All region refreshes completed.")
    return is_admin, configured_token


def _has_cached_public_data(conn) -> bool:
    return any(count_rows(conn, table_name) > 0 for table_name in ("sensor_readings", "weather_snapshots", "risk_snapshots"))


def _first_run_onboarding(conn, configured_token: str | None) -> None:
    if _has_cached_public_data(conn):
        return
    st.warning(
        "No cached public data is available yet. Run `python scripts/ingest.py` before demoing all regions, "
        "or configure `ADMIN_TOKEN` and refresh from the sidebar."
    )
    if configured_token is None:
        st.caption("Because `ADMIN_TOKEN` is unset, sidebar refresh actions are disabled in this session.")
    else:
        st.caption("Enter the configured admin token in the sidebar, then refresh one region or all regions.")


def _map(region: dict, observations: list[dict]) -> None:
    st.subheader("Map")
    station = pd.DataFrame(
        [
            {
                "latitude": region["latitude"],
                "longitude": region["longitude"],
                "name": f"USGS {region['usgs_site_no']}",
                "kind": "USGS station",
            }
        ]
    )
    obs = pd.DataFrame(observations)
    layers = [
        pdk.Layer(
            "ScatterplotLayer",
            station,
            get_position="[longitude, latitude]",
            get_fill_color="[30, 90, 200, 220]",
            get_radius=80,
            pickable=True,
        )
    ]
    if not obs.empty:
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                obs,
                get_position="[longitude, latitude]",
                get_fill_color="[220, 95, 45, 190]",
                get_radius=45,
                pickable=True,
            )
        )
    st.pydeck_chart(
        pdk.Deck(
            initial_view_state=pdk.ViewState(
                latitude=region["latitude"],
                longitude=region["longitude"],
            zoom=region["map_zoom"],
            pitch=0,
        ),
            map_style="https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            layers=layers,
            tooltip={"text": "{name}\n{timestamp_utc}\nRunoff: {runoff_present}"},
        )
    )
    st.caption("Legend: blue marker = USGS station; orange markers = field observations.")
    if obs.empty:
        st.info("No field observations have been logged for this region yet.")
    else:
        st.dataframe(obs, width="stretch", hide_index=True)


def render_dashboard(conn, region: dict) -> None:
    st.title(APP_NAME)
    st.subheader(region["display_name"])
    st.write(region["description"])
    st.caption(
        f"{region['waterway_name']} | {region['city']}, {region['state']} | "
        f"USGS site {region['usgs_site_no']} | Coordinates are editable configuration values."
    )

    latest_risk = get_latest_risk(conn, region["id"])
    latest_weather = get_latest_weather(conn, region["id"])
    latest_readings = get_latest_sensor_readings(conn, region["id"])
    latest_prediction = get_latest_ml_prediction(conn, region["id"])
    region_note = get_region_data_note(region.get("slug"))

    data_quality_banner(latest_weather, latest_readings, region_note)
    cols = st.columns(4)
    category = latest_risk["category"] if latest_risk else "Insufficient Data"
    with cols[0]:
        st.metric("Current risk", risk_metric_label(latest_risk))
        risk_badge(category)
    with cols[1]:
        st.metric("Rule-based score", "Unavailable" if not latest_risk or latest_risk["score"] is None else f"{latest_risk['score']:.1f}")
    with cols[2]:
        st.metric("Confidence", "Unavailable" if not latest_risk else f"{latest_risk['confidence']:.0%}")
    with cols[3]:
        st.metric("Available weight", "Unavailable" if not latest_risk else f"{latest_risk['available_weight']:.0f}/100")
        st.caption("How much of the 100-point formula has usable inputs.")

    st.subheader("Why This Score?")
    if latest_risk:
        st.write(latest_risk["explanation"])
        component_breakdown(latest_risk)
    else:
        st.info(
            "No rule-based score is available yet. Use an admin token to refresh this region, "
            "or run `python scripts/ingest.py` locally to preload all predefined regions."
        )

    st.subheader("Current Conditions")
    left, right = st.columns(2)
    with left:
        st.write("Latest stream readings")
        if latest_readings:
            st.dataframe(pd.DataFrame(latest_readings), width="stretch", hide_index=True)
        else:
            if region_note:
                st.info(f"No current USGS readings are cached for this region. {region_note}")
            else:
                st.info("No current USGS readings are cached for this region yet.")
    with right:
        st.write("Latest weather/rainfall snapshot")
        if latest_weather:
            st.dataframe(pd.DataFrame([latest_weather]), width="stretch", hide_index=True)
        else:
            st.info("No weather snapshot cached yet.")

    recent_risk = get_recent_table(conn, "risk_snapshots", region["id"])
    risk_chart(recent_risk, chart_key=f"dashboard-risk-chart-{region['id']}")
    ml_research_panel(latest_prediction, latest_risk)
    _map(region, get_field_observations(conn, region["id"]))


def render_trends(conn, region: dict) -> None:
    st.header("Trends")
    range_label = st.selectbox("Time range", ["24 hours", "7 days", "30 days"], index=1)
    hours = {"24 hours": 24, "7 days": 24 * 7, "30 days": 24 * 30}[range_label]
    sensor_frame = get_recent_table(conn, "sensor_readings", region["id"], hours=hours)
    risk_frame = get_recent_table(conn, "risk_snapshots", region["id"], hours=hours)
    weather_frame = get_recent_table(conn, "weather_snapshots", region["id"], hours=hours)
    prediction_frame = get_recent_table(conn, "ml_predictions", region["id"], hours=hours)

    parameter_options = ["All"]
    if not sensor_frame.empty:
        parameter_options += sorted(sensor_frame["parameter_code"].dropna().unique().tolist())
    parameter = st.selectbox("Parameter", parameter_options)

    risk_chart(risk_frame, chart_key=f"trends-risk-chart-{region['id']}")
    sensor_chart(sensor_frame, None if parameter == "All" else parameter, chart_key=f"trends-sensor-chart-{region['id']}")
    rainfall_chart(weather_frame, chart_key=f"trends-rainfall-chart-{region['id']}")

    export_tabs = st.expander("CSV exports")
    with export_tabs:
        if not risk_frame.empty:
            st.download_button("Download risk history CSV", risk_frame.to_csv(index=False), "risk_history.csv", "text/csv")
        if not sensor_frame.empty:
            st.download_button("Download sensor readings CSV", sensor_frame.to_csv(index=False), "sensor_readings.csv", "text/csv")
        if not weather_frame.empty:
            st.download_button("Download weather snapshots CSV", weather_frame.to_csv(index=False), "weather_snapshots.csv", "text/csv")
        if not prediction_frame.empty:
            st.download_button("Download ML predictions CSV", prediction_frame.to_csv(index=False), "ml_predictions.csv", "text/csv")

    if not prediction_frame.empty:
        y_cols = [
            column
            for column in ["anomaly_score", "forecast_probability"]
            if column in prediction_frame and prediction_frame[column].notna().any()
        ]
        if y_cols:
            fig = px.line(prediction_frame, x="timestamp_utc", y=y_cols, markers=True, title="ML Research Signals")
            st.plotly_chart(fig, width="stretch", key=f"trends-ml-chart-{region['id']}")
    else:
        st.info("No ML prediction history is available for this range.")


def render_field_log(conn, region: dict, is_admin: bool) -> None:
    st.header("Field Log")
    st.write("Authorized admins can log moderated field observations for the selected region.")
    region_tz = ZoneInfo(region["timezone"])
    now_local = datetime.now(region_tz)

    with st.form("field_observation_form"):
        date_value = st.date_input("Observation date", value=now_local.date(), disabled=not is_admin)
        time_value = st.time_input("Observation time", value=time(hour=now_local.hour, minute=now_local.minute), disabled=not is_admin)
        observer_name = st.text_input("Observer name (optional)", disabled=not is_admin)
        cols = st.columns(2)
        latitude = cols[0].number_input("Latitude", value=float(region["latitude"]), format="%.6f", disabled=not is_admin)
        longitude = cols[1].number_input("Longitude", value=float(region["longitude"]), format="%.6f", disabled=not is_admin)
        water_clarity_score = st.selectbox("Water clarity score (optional)", [None, 1, 2, 3, 4, 5], disabled=not is_admin)
        visual_turbidity_score = st.selectbox("Visual turbidity score (optional)", [None, 1, 2, 3, 4, 5], disabled=not is_admin)
        runoff_present = st.radio("Runoff present", [True, False], format_func=lambda value: "Yes" if value else "No", disabled=not is_admin)
        odor_present = st.checkbox("Odor present", disabled=not is_admin)
        trash_or_debris_present = st.checkbox("Trash or debris present", disabled=not is_admin)
        notes = st.text_area("Notes (1000 characters max)", max_chars=1000, disabled=not is_admin)
        photo_url = st.text_input("Photo URL (optional)", disabled=not is_admin)
        submitted = st.form_submit_button("Save Observation", disabled=not is_admin)

    if not is_admin:
        st.info("Enter the configured admin token in the sidebar to add field observations.")
    elif submitted:
        local_dt = datetime.combine(date_value, time_value, tzinfo=region_tz)
        payload = {
            "timestamp_utc": local_dt.astimezone(timezone.utc),
            "observer_name": observer_name,
            "latitude": latitude,
            "longitude": longitude,
            "water_clarity_score": water_clarity_score,
            "visual_turbidity_score": visual_turbidity_score,
            "runoff_present": runoff_present,
            "odor_present": odor_present,
            "trash_or_debris_present": trash_or_debris_present,
            "notes": notes,
            "photo_url": photo_url,
        }
        try:
            insert_field_observation(conn, region["id"], payload)
            st.success("Observation saved.")
        except ValidationError as exc:
            LOGGER.info("Invalid field observation: %s", exc)
            st.error(str(exc))
        except Exception:
            LOGGER.exception("Failed to save field observation")
            st.error("Could not save the observation. Please check the form and try again.")

    observations = get_field_observations(conn, region["id"])
    if observations:
        st.dataframe(pd.DataFrame(observations), width="stretch", hide_index=True)
    else:
        st.info("No observations have been saved for this region yet.")


def render_methodology(regions: list[dict]) -> None:
    st.header("Methodology")
    st.write(
        "This public dashboard combines public USGS instantaneous stream data with Open-Meteo precipitation "
        "and forecast data for predefined U.S. urban waterways. It calculates a transparent rule-based "
        "runoff-risk score and stores local snapshots for repeatable analysis."
    )
    st.subheader("Supported Regions")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "Region": region["display_name"],
                    "USGS site": region["usgs_site_no"],
                    "Timezone": region["timezone"],
                    "Known data limitation": get_region_data_note(region.get("slug")) or "None currently documented.",
                }
                for region in regions
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    st.subheader("Rule-Based Risk Formula")
    st.markdown(
        """
- Recent rainfall: 40 points, using 1-hour, 6-hour, and 24-hour precipitation.
- Stream/gage condition: 30 points, comparing discharge or gage height to the recent local baseline.
- Turbidity/water-quality proxy: 20 points, when turbidity is available.
- Forecast rainfall: 10 points, using the next 6 hours of precipitation forecast.

If less than 50 percent of component weight is available, the category is **Insufficient Data**.
Confidence is the available component weight, reduced when stream or turbidity baselines have fewer than 20 readings.
        """
    )
    st.subheader("ML / Anomaly Layer")
    st.write(
        "The ML layer is optional and research-oriented. Feature rows align rainfall, stream readings, water-quality "
        "proxies, percentiles, and rise rates. Anomaly detection can use rolling z-scores, median absolute deviation "
        "style baselines, or Isolation Forest. Supervised forecasting requires at least 500 aligned samples and at "
        "least 20 positive/high-risk events."
    )
    st.subheader("Safety Disclaimer")
    st.warning(
        "This dashboard is for educational and community awareness purposes only. It does not determine whether water "
        "is safe to drink, touch, swim in, or use. Do not enter fast-moving water or unsafe areas to collect observations."
    )
    st.subheader("Future Improvements")
    st.write(
        "Phase 2 can add deeper backtesting, model cards, richer exportable reports, API migration to newer USGS "
        "Water Data APIs, and portfolio-ready demo datasets."
    )


def render_ml_evaluation(conn, region: dict) -> None:
    st.header("ML Evaluation")
    features = get_ml_feature_dataframe(conn, region["id"])
    latest_model = get_latest_model_run(conn, region["id"])
    if features.empty:
        st.info("No ML feature rows have been generated for this region yet.")
    else:
        positives = count_positive_targets(features)
        cols = st.columns(3)
        cols[0].metric("Feature rows", len(features))
        cols[1].metric("Positive/high-risk events", positives)
        cols[2].metric("Forecast-ready", "Yes" if len(features) >= 500 and positives >= 20 else "No")

    if latest_model is None:
        st.info("No trained model is available for this region yet. The app is currently using the transparent rule-based risk score.")
        return

    st.subheader("Latest Model Run")
    st.dataframe(pd.DataFrame([latest_model]), width="stretch", hide_index=True)
    if latest_model.get("metrics_json"):
        st.subheader("Metrics")
        st.json(json.loads(latest_model["metrics_json"]))
    if latest_model.get("feature_importance_json"):
        st.subheader("Feature Importance")
        st.json(json.loads(latest_model["feature_importance_json"]))


def main() -> None:
    st.set_page_config(page_title=APP_NAME, layout="wide")
    conn = get_connection()
    regions = get_active_regions(conn)
    if not regions:
        st.error("No active regions are configured.")
        return
    region = _selected_region(conn, regions)
    is_admin, configured_token = _admin_controls(region)
    _first_run_onboarding(conn, configured_token)

    tabs = st.tabs(["Dashboard", "Trends", "Field Log", "Methodology", "ML Evaluation"])
    with tabs[0]:
        render_dashboard(conn, region)
    with tabs[1]:
        render_trends(conn, region)
    with tabs[2]:
        render_field_log(conn, region, is_admin)
    with tabs[3]:
        render_methodology(regions)
    with tabs[4]:
        render_ml_evaluation(conn, region)


if __name__ == "__main__":
    main()
