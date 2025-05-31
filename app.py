import pandas as pd
import numpy as np

# Create a date range for the past 30 days at exactly 7PM
date_range = pd.date_range(end=pd.Timestamp.today(), periods=30, freq='D').normalize() + pd.Timedelta(hours=19)

# Generate fake sensor data
data = {
    "timestamp": date_range,
    "pH": np.random.uniform(5.5, 8.5, size=30).round(2),
    "TDS": np.random.uniform(300, 900, size=30).round(1),
    "Temperature (°C)": np.random.uniform(20, 35, size=30).round(1),
    "Light (LDR)": np.random.randint(100, 1000, size=30),
    "Distance (cm)": np.random.uniform(10, 50, size=30).round(1),
    "LED Relay Status": np.random.choice([0, 1], size=30)
}

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.dataframe(df)

