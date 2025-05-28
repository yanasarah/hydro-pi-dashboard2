import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Hydro-Pi Smart Farming Dashboard", layout="wide")

st.title("🌱 Hydro-Pi Smart Farming Dashboard")
st.markdown("""
Monitor your smart hydroponic system in real-time.  
Upload sensor data to see insights, warnings, and trends.
""")

# File uploader
uploaded_file = st.file_uploader("📤 Upload your sensor CSV file", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = [col.strip() for col in df.columns]

    time_col = df.columns[0]
    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except Exception as e:
        st.error(f"❌ Couldn't parse time column: {e}")
        st.stop()

    sensor_cols = df.columns[1:]

    # Remove rows with missing time
    df = df.dropna(subset=[time_col])

    if df.empty:
        st.warning("⚠️ No valid data after filtering timestamps.")
        st.stop()

    # Time Range Slider (SAFE)
    min_time = df[time_col].min()
    max_time = df[time_col].max()

    if min_time == max_time:
        st.warning("⚠️ Only one timestamp in the data.")
        filtered_df = df
    else:
        time_range = st.slider(
            "⏱️ Select Time Range", 
            min_value=min_time, 
            max_value=max_time, 
            value=(min_time, max_time), 
            format="YYYY-MM-DD – HH:mm"
        )
        filtered_df = df[(df[time_col] >= time_range[0]) & (df[time_col] <= time_range[1])]

    # Sensor Warnings
    st.subheader("⚠️ Sensor Warnings")
    if "pH" in sensor_cols:
        ph_data = pd.to_numeric(filtered_df["pH"], errors="coerce")
        if ph_data.max() > 7.5:
            st.warning("⚠️ High pH level detected (> 7.5)")
        elif ph_data.min() < 5.5:
            st.warning("⚠️ Low pH level detected (< 5.5)")

    if "Water Level (cm)" in sensor_cols:
        wl_data = pd.to_numeric(filtered_df["Water Level (cm)"], errors="coerce")
        if wl_data.min() < 8:
            st.warning("⚠️ Low Water Level detected (< 8 cm)")

    # Charts
    st.subheader("📊 Sensor Data Trends")
    for sensor in sensor_cols:
        y_data = pd.to_numeric(filtered_df[sensor], errors="coerce")
        fig = px.line(filtered_df, x=time_col, y=y_data, markers=True,
                      title=f"{sensor} Trend", labels={time_col: "Time", sensor: sensor})
        st.plotly_chart(fig, use_container_width=True)

        col1, col2, col3 = st.columns(3)
        col1.metric("Avg", f"{y_data.mean():.2f}")
        col2.metric("Max", f"{y_data.max():.2f}")
        col3.metric("Min", f"{y_data.min():.2f}")
        st.markdown("---")
else:
    st.info("Please upload a CSV file. You can use this [sample CSV](sandbox:/mnt/data/hydro_pi_sample_data.csv) for testing.")
