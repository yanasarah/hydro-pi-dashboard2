import streamlit as st
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np


# Set Streamlit page config
st.set_page_config(page_title="Hydro-Pi Smart Dashboard", layout="wide")

# Sidebar navigation menu
with st.sidebar:
    selected = option_menu(
        menu_title="🌿 Hydro-Pi Dashboard",  # Sidebar title
        options=["Home", "Environment Monitor", "Growth Consistency", "Insights", "Contact"],
        icons=["house", "bar-chart", "activity", "lightbulb", "envelope"],
        menu_icon="cast",
        default_index=0
    )
#=======FOR HOME================
if selected == "Home":
    st.title("🌱 Welcome to Hydro-Pi Smart Farming Dashboard")
    st.write("Upload your sensor data CSV to view predictions of plant growth.")

    uploaded_file = st.file_uploader("📁 Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Raw Uploaded Data")
        st.dataframe(df)

        # Simulate plant_growth (if not already there)
        if 'plant_growth' not in df.columns:
            try:
                df['plant_growth'] = (
                    0.3 * df['temperature'] +
                    0.2 * df['tds'] +
                    0.2 * df['ldr'] -
                    0.2 * abs(df['ph'] - 6.5) +  # Optimal pH is ~6.5
                    0.1 * df['distance']
                )
            except Exception as e:
                st.error(f"Missing expected sensor columns. Please check your file. Error: {e}")
                st.stop()

        # Drop non-numeric or unused columns if any
       # Drop 'plant_growth' and non-numeric columns (like datetime)
X = df.drop(columns=['plant_growth'])
X = X.select_dtypes(include=['float64', 'int64'])

        y = df['plant_growth']

        # Impute and scale
        imputer = SimpleImputer(strategy="mean")
        scaler = StandardScaler()

        try:
            X_imputed = imputer.fit_transform(X)
            X_scaled = scaler.fit_transform(X_imputed)
        except Exception as e:
            st.error(f"Preprocessing error: {e}")
            st.stop()

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict
        y_pred = model.predict(X_test)

        st.subheader("📊 Machine Learning Prediction Results")
        results_df = pd.DataFrame({
            "Actual Growth": y_test.values,
            "Predicted Growth": y_pred
        })
        st.dataframe(results_df)

        # Display cleaned input table
        st.subheader("✅ Cleaned Data with Simulated Growth")
        df['predicted_growth'] = model.predict(scaler.transform(imputer.transform(X)))
        st.dataframe(df)

        # Optional: Show model performance
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.metric("R² Score", round(r2, 3))
        st.metric("Mean Squared Error", round(mse, 3))
