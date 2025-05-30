import streamlit as st
import pandas as pd

file = st.file_uploader("📤 Upload CSV", type=["csv"])

if file:
    df = pd.read_csv(file)

    # Ensure 'timestamp' is in datetime format
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors='coerce')
        df.dropna(subset=["timestamp"], inplace=True)
        df.sort_values("timestamp", inplace=True)

        if not df.empty:
            min_time = df["timestamp"].min()
            max_time = df["timestamp"].max()

            # Only show slider if timestamps are valid
            if pd.notna(min_time) and pd.notna(max_time):
                time_range = st.slider(
                    "⏱️ Select Time Range",
                    min_value=min_time,
                    max_value=max_time,
                    value=(min_time, max_time),
                    format="MM/DD/YY - HH:mm"
                )

                df = df[(df["timestamp"] >= time_range[0]) & (df["timestamp"] <= time_range[1])]
            else:
                st.warning("❗ Timestamps could not be parsed.")
        else:
            st.warning("❗ No data available after cleaning timestamps.")
    else:
        st.error("❌ Column 'timestamp' not found in uploaded CSV.")
