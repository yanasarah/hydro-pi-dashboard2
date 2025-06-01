import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# Save uploaded CSV globally
if "csv_data" not in st.session_state:
    st.session_state.csv_data = None
if "cleaned_data" not in st.session_state:
    st.session_state.cleaned_data = None

# Sidebar menu
with st.sidebar:
    selected = option_menu(
        menu_title="MAIN MENU",
        options=["home", "project", "contact"],
        icons=["house", "gear", "envelope"],
        menu_icon="cast",
        default_index=0
    )

# Home page
if selected == "home":
    st.title("🌱 Welcome to Hydro-Pi Smart Farming Dashboard")
    st.write("Upload your sensor CSV file below to begin analysis and ML-based data cleaning:")

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            # Load CSV
            df = pd.read_csv(uploaded_file)
            st.session_state.csv_data = df

            st.success("✅ File uploaded successfully!")
            st.subheader("📋 Original Data")
            st.dataframe(df)

            # Machine learning-based data cleaning
            numeric_df = df.select_dtypes(include=['float64', 'int64']).copy()

            # Impute missing values using mean strategy
            imputer = SimpleImputer(strategy='mean')
            imputed_data = imputer.fit_transform(numeric_df)

            # Normalize / scale the data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(imputed_data)

            # Create cleaned DataFrame
            cleaned_df = pd.DataFrame(scaled_data, columns=numeric_df.columns)
            st.session_state.cleaned_data = cleaned_df

            st.subheader("🧹 Cleaned Data (ML-based preprocessing)")
            st.dataframe(cleaned_df)

        except Exception as e:
            st.error(f"❌ Error reading or processing file: {e}")

# Project and contact pages remain unchanged...
elif selected == "project":
    st.title("📊 Sensor Data Charts")

    if st.session_state.csv_data is not None:
        df = st.session_state.csv_data
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])

        sensor_columns = ['LED Relay Status', 'pH', 'TDS', 'Temperature (°C)', 'Distance (cm)']
        for sensor in sensor_columns:
            if sensor in df.columns:
                st.subheader(f"{sensor} Reading")
                fig = px.line(df, x='timestamp' if 'timestamp' in df.columns else df.index,
                              y=sensor, markers=True, title=f"{sensor} Over Time")
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning(f"⚠️ '{sensor}' data not found in the uploaded CSV.")
    else:
        st.warning("⚠️ Please upload a CSV file first on the **Home** page.")

elif selected == "contact":
    st.title("📞 This is the Contact Page")

