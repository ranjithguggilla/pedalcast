"""Streamlit scoring UI. Run with: streamlit run src/pedalcast/app.py"""

from __future__ import annotations

import streamlit as st

from pedalcast.app_logic import WEATHER_LABELS, explain, load_bundle, predict_demand

SEASONS = {"winter": 1, "spring": 2, "summer": 3, "fall": 4}
WEEKDAYS = {"Sun": 0, "Mon": 1, "Tue": 2, "Wed": 3, "Thu": 4, "Fri": 5, "Sat": 6}


def main() -> None:
    st.set_page_config(page_title="PedalCast", page_icon="bike")
    st.title("PedalCast")
    st.caption("Forecast the next hour of bike-share demand so the fleet is where riders are.")

    try:
        bundle = load_bundle()
    except FileNotFoundError:
        st.error("No trained model found. Run `python -m pedalcast.train` first.")
        return

    col1, col2 = st.columns(2)
    with col1:
        season = SEASONS[st.selectbox("Season", list(SEASONS))]
        month = st.slider("Month", 1, 12, 6)
        hour = st.slider("Hour of day", 0, 23, 8)
        weekday = WEEKDAYS[st.selectbox("Day of week", list(WEEKDAYS), index=1)]
    with col2:
        weather = st.selectbox(
            "Weather", list(WEATHER_LABELS), format_func=lambda k: WEATHER_LABELS[k]
        )
        temp = st.slider("Temperature (normalised)", 0.0, 1.0, 0.5, 0.02)
        humidity = st.slider("Humidity (normalised)", 0.0, 1.0, 0.6, 0.02)
        windspeed = st.slider("Wind speed (normalised)", 0.0, 1.0, 0.2, 0.02)

    workingday = 0 if weekday in (0, 6) else 1
    inputs = {
        "season": season, "month": month, "hour": hour, "weekday": weekday,
        "workingday": workingday, "weather": weather, "temp": temp,
        "humidity": humidity, "windspeed": windspeed, "year": 1,
    }

    demand = predict_demand(bundle, inputs)
    st.metric("Predicted bikes rented this hour", demand)
    st.write("Why:")
    for reason in explain(inputs):
        st.write(f"- {reason}")


if __name__ == "__main__":
    main()
