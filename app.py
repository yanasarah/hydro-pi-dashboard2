import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

st.title("📈 Plant Sensor Dashboard")

# File uploader
uploaded_file = st.file_uploader("Upload your sensor CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Data Preview")
    st.write(df.head())

    # Select a column to plot
    sensor_column = st.selectbox("Choose a sensor column to plot", df.columns[1:])

    # Plot
    st.subheader(f"Line Chart for: {sensor_column}")
    fig, ax = plt.subplots()
    ax.plot(df[df.columns[0]], df[sensor_column], marker='o')
    ax.set_xlabel(df.columns[0])
    ax.set_ylabel(sensor_column)
    st.pyplot(fig)
