import streamlit as st
import pandas as pd
import plotly.express as px
import datetime

st.set_page_config(layout="wide")

# ----- CONFIGURE PAGE -----
st.set_page_config(page_title="Hydro-Pi Dashboard", layout="wide")

# ----- CUSTOM CSS TO TIGHTEN LAYOUT -----
st.markdown("""
    <style>
        .block-container {
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1rem;
            padding-right: 1rem;
        }
        [data-testid="stVerticalBlock"] {
            gap: 0.5rem;
        }
    </style>
    """, unsafe_allow_html=True)

# ----- TITLE -----
st.title("🌿 Hydro-Pi Smart Farming Dashboard")

# ----- UPLOAD CSV -----
st.sidebar.header("📁 Upload Your CSV")
file = st.sidebar.file_uploader("Choose a CSV file", type="csv")

if file:
    df = pd.read_csv(file)

    # Ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
    df.dropna(subset=["timestamp"], inplace=True)
    df.sort_values("timestamp", inplace=True)

    # ----- TIME RANGE SLIDER -----
    min_time = df["timestamp"].min()
    max_time = df["timestamp"].max()

    # Convert Timestamp objects to Python datetime objects
    min_time_dt = min_time.to_pydatetime()
    max_time_dt = max_time.to_pydatetime()

    time_range = st.slider("⏱️ Select Time Range", min_value=min_time_dt, max_value=max_time_dt,
                           value=(min_time_dt, max_time_dt), format="MM/DD/YY - HH:mm")

    # Convert the selected time range back to Timestamp for filtering
    start_time = pd.Timestamp(time_range[0])
    end_time = pd.Timestamp(time_range[1])

    df = df[(df["timestamp"] >= start_time) & (df["timestamp"] <= end_time)]

    sensor_cols = df.columns.drop("timestamp")

    # ----- SUMMARY METRICS -----
    st.subheader("📊 Sensor Metrics Summary")
    cols = st.columns(len(sensor_cols))
    for i, sensor in enumerate(sensor_cols):
        avg_val = df[sensor].mean()
        cols[i].metric(sensor, f"{avg_val:.2f}")

    # ----- WARNING INDICATORS -----
    st.subheader("⚠️ Warnings")
    with st.expander("Click to view warnings"):
        for sensor in sensor_cols:
            if "pH" in sensor and df[sensor].max() > 7.5:
                st.warning(f"{sensor}: High pH Level Detected!")
            if "TDS" in sensor and df[sensor].mean() > 1000:
                st.warning(f"{sensor}: High TDS Level!")
            if "Temperature" in sensor and df[sensor].mean() > 35:
                st.warning(f"{sensor}: High Temperature!")

    # ----- SENSOR CHARTS -----
    st.subheader("📈 Sensor Trends")
    for sensor in sensor_cols:
        fig = px.line(df, x="timestamp", y=sensor, title=f"{sensor} Over Time")
        st.plotly_chart(fig, use_container_width=True)

else:
    st.info("Please upload a CSV file to display the dashboard.")
