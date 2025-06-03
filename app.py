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

    

# Home Section - Upload CSV & Predict
if selected == "Home":
    st.title("🌱 Welcome to Hydro-Pi Smart Farming Dashboard")
    st.markdown("Upload your environmental sensor data to predict plant growth trends.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.subheader("Raw Data")
        st.dataframe(df)

        # Select only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        # Clean the data
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X = df_numeric.copy()

        # Simulate plant_growth for training
        np.random.seed(42)
        X['plant_growth'] = (
            0.2 * X.get('pH', 0) +
            0.25 * X.get('TDS', 0) +
            0.2 * X.get('temperature', 0) +
            0.15 * X.get('ldr', 0) +
            0.1 * X.get('distance', 0) +
            0.1 * X.get('LED Relay Status', 0) +
            np.random.normal(0, 0.5, size=len(X))
        )

        # Drop rows with all NaNs
        X = X.dropna(how='all')

        # Impute and scale
        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        # Split features and target
        y = X['plant_growth'].values
        X_features = X.drop(columns=['plant_growth'])
        X_train, X_test, y_train, y_test = train_test_split(
            X_features, y, test_size=0.2, random_state=42
        )

        # Train model
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        st.subheader("Cleaned and Enriched Data")
        st.dataframe(X.assign(plant_growth=np.round(y, 2)))

        st.subheader("📈 ML Model Performance")
        st.write(f"Mean Squared Error on Test Set: {mse:.3f}")

        st.subheader("🌿 Predicted Growth (Sample)")
        pred_df = pd.DataFrame({
            'Actual': np.round(y_test, 2),
            'Predicted': np.round(predictions, 2)
        })
        st.dataframe(pred_df.head(10))

# Environment Monitor Section
elif selected == "Environment Monitor":
    st.title("📊 Environmental Monitoring")

    if uploaded_file is not None:
        st.markdown("Visualizing sensor trends from your uploaded data.")
        for col in ['pH', 'TDS', 'Temperature', 'LDR', 'Distance (cm)']:
            if col in df.columns:
                st.subheader(f"{col} Trend")
                st.line_chart(df[col])
    else:
        st.warning("⚠️ Please upload a CSV file from the Home section first.")

