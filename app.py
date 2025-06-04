import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Set Streamlit page configuration
st.set_page_config(page_title="Hydro-Pi Smart Dashboard", layout="wide")
#=================NAVIGATION==============================
# Sidebar Navigation
with st.sidebar:
    selected = option_menu(
        menu_title="🌿 Hydro-Pi Dashboard",
        options=["Home", "Environment Monitor", "Growth Consistency", "Insights", "Contact"],
        icons=["house", "bar-chart", "activity", "lightbulb", "envelope"],
        menu_icon="cast",
        default_index=0
    )
#=====================HOME=========================
# HOME SECTION
if selected == "Home":
    st.title("🌱 Welcome to Hydro-Pi Smart Farming Dashboard")
    st.markdown("Upload your environmental sensor data to predict plant growth trends.")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file  # Save file
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df  # Save DataFrame to session

        st.subheader("📂 Raw Data")
        st.dataframe(df)

        # Only numeric columns
        df_numeric = df.select_dtypes(include=[np.number])

        # Prepare and simulate plant_growth
        X = df_numeric.copy()

        np.random.seed(42)
        X['plant_growth'] = (
            0.2 * X.get('pH', 0) +
            0.25 * X.get('TDS', 0) +
            0.2 * X.get('Temperature', 0) +
            
            0.1 * X.get('Distance (cm)', 0) +
            0.1 * X.get('LED', 0) +
            np.random.normal(0, 0.5, size=len(X))
        )

        X = X.dropna(how='all')

        # ML Preprocessing
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        y = X['plant_growth'].values
        X_features = X.drop(columns=['plant_growth'])
        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

        # Train RandomForest
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        # Display Cleaned Data
        st.subheader("🧹 Cleaned & Enriched Data")
        st.dataframe(X.assign(plant_growth=np.round(y, 2)))

        # ML Performance
        st.subheader("📈 ML Model Performance")
        st.write(f"Mean Squared Error: {mse:.3f}")

        # Sample Predictions
        st.subheader("🌿 Predicted Growth (Sample)")
        pred_df = pd.DataFrame({
            'Actual': np.round(y_test, 2),
            'Predicted': np.round(predictions, 2)
        })
        st.dataframe(pred_df.head(10))

#=============BAHAGIAN ENVIRONMENT=============================
# ENVIRONMENT MONITOR SECTION
elif selected == "Environment Monitor":
    st.title("📊 Environmental Monitoring")
    
    if 'df' in st.session_state:
        df = st.session_state.df
        st.markdown("Visualizing trends from your uploaded data.")

        for col in ['pH', 'TDS', 'Temperature', 'LED', 'Distance (cm)']:
            if col in df.columns:
                st.subheader(f"{col} Trend")
                st.line_chart(df[col])
    else:
        st.warning("⚠️ Please upload a CSV file from the Home section first.")

#===================SECTION GROWTH===============
# GROWTH CONSISTENCY SECTION (Placeholder for now)
elif selected == "Growth Consistency":
    st.title("🌾 Growth Consistency Analysis")

    if 'df' in st.session_state:
        df = st.session_state.df

        # Recalculate simulated plant_growth
        df_numeric = df.select_dtypes(include=[np.number]).copy()

        df_numeric['plant_growth'] = (
            0.2 * df_numeric.get('pH', 0) +
            0.25 * df_numeric.get('TDS', 0) +
            0.2 * df_numeric.get('Temperature', 0) +
            0.1 * df_numeric.get('Distance (cm)', 0) +
            0.1 * df_numeric.get('LED', 0) +
            np.random.normal(0, 0.5, size=len(df_numeric))
        )

        st.subheader("📊 Environmental Stability (Standard Deviation)")
        env_cols = ['pH', 'TDS', 'Temperature', 'LED', 'Distance (cm)']
        existing_env_cols = [col for col in env_cols if col in df_numeric.columns]
missing_cols = [col for col in env_cols if col not in df_numeric.columns]

if missing_cols:
    st.warning(f"⚠️ These columns are missing from your data: {', '.join(missing_cols)}")

env_stability = df_numeric[existing_env_cols].std().round(2)
st.write(env_stability)


        st.subheader("🌿 Growth Consistency")
        growth_std = df_numeric['plant_growth'].std()
        st.metric("Growth Std Deviation", f"{growth_std:.2f}")

        # Alert based on threshold
        if growth_std > 1.5:
            st.warning("⚠️ High variability in plant growth. Consider stabilizing environmental conditions.")
        else:
            st.success("✅ Growth conditions are stable.")

        # Line charts
        st.subheader("📈 Environmental Trends vs. Growth")
        for col in env_cols:
            if col in df_numeric.columns:
                st.line_chart(df_numeric[[col, 'plant_growth']])
    else:
        st.warning("⚠️ Please upload a CSV file from the Home section first.")



# INSIGHTS SECTION (Placeholder)
elif selected == "Insights":
    st.title("💡 Insights & Recommendations")
    st.info("Coming soon: Smart suggestions based on plant conditions.")


# CONTACT SECTION
elif selected == "Contact":
    st.title("📞 Contact Us")
    st.markdown("""
        **Hydro-Pi Team**  
        📧 Email: support@hydro-pi.local  
        🌍 Website: [www.hydro-pi.local](#)
    """)

