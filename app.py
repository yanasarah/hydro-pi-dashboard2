import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from datetime import datetime

# Set Streamlit page config
st.set_page_config(page_title="Hydro-Pi Smart Dashboard", layout="wide")

#==================== FONT & STYLE ====================
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
        .stApp {
            background-color: #f5fff5;
            color: #006400;
            font-family: 'Poppins', sans-serif;
        }
        [data-testid="stSidebar"] {
            background-color: #c0ebc0 !important;
            min-width: 210px !important;
            max-width: 230px !important;
        }
        [data-testid="stSidebar"] * {
            color: #003300 !important;
            font-size: 14px !important;
        }
        .stMetric label {
            color: #006400 !important;
        }
        .stDataFrame thead tr th {
            background-color: #e0f5e0 !important;
            color: #006400 !important;
        }
    </style>
""", unsafe_allow_html=True)

#================= NAVIGATION =========================
with st.sidebar:
    selected = option_menu(
        menu_title="🌿 Hydro-Pi Dashboard",
        options=["Home", "Historical Data", "Environment Monitor", "Growth Consistency", "Insights", "Contact"],
        icons=["house", "navigation", "bar-chart", "activity", "lightbulb", "envelope"],
        menu_icon="cast",
        default_index=0
    )

#================= HOME SECTION =======================
if selected == "Home":
    st.markdown("""
        <div style="padding: 2rem; background: linear-gradient(to right, #bdfcc9, #e0ffe0); border-radius: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); text-align: center;">
            <h1 style="color: #2e8b57;">🌱 Welcome to Hydro-Pi Smart Farming</h1>
            <p style="color: #4d774e; font-size: 18px;">Monitor. Predict. Grow smarter 🌿</p>
        </div>
        <br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <div style='background-color: #e6ffe6; border-left: 5px solid #66bb66; padding: 1rem; border-radius: 12px;'>
                🌿 <em>“Grow your health, grow a garden.”</em>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background-color: #ffffff; border: 2px solid #cceccc; border-radius: 12px; padding: 1rem; text-align: center;'>
                <h4>Your Plant:</h4>
                <p style='font-weight: bold; color: #2e7d32;'>🥬 Spinach</p>
                <img src="https://www.pngmart.com/files/13/Spinach-PNG-Transparent-Image.png" alt="Spinach" width="100">
            </div>
        """, unsafe_allow_html=True)

    st.markdown(f"""
        <br>
        <div style="background-color: #e0f5e9; padding: 1.5rem; border-radius: 15px; text-align: center;">
            <h4>📅 Today</h4>
            <p style="font-size: 20px;">{datetime.now().strftime("%A, %d %B %Y")}</p>
        </div>
    """, unsafe_allow_html=True)

#================ HISTORICAL DATA ======================
elif selected == "Historical Data":
    st.markdown("<h1 style='color:#2e8b57;'>📁 Historical Data</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("📤 Upload CSV Sensor Data", type=["csv"])

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.subheader("📂 Raw Data")
        st.dataframe(df.head(20))

        # Prepare for ML
        df_numeric = df.select_dtypes(include=[np.number]).copy()
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

        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        with st.spinner("🧠 Training prediction model..."):
            X_filled = imputer.fit_transform(X)
            X_scaled = scaler.fit_transform(X_filled)

            y = X['plant_growth'].values
            X_features = X.drop(columns=['plant_growth'])
            X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

            model = RandomForestRegressor(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            mse = mean_squared_error(y_test, predictions)

        st.subheader("🧹 Cleaned & Enriched Data")
        st.dataframe(X.assign(plant_growth=np.round(y, 2)).head(20))

        st.subheader("📈 Model Performance")
        st.metric("Mean Squared Error", f"{mse:.3f}")

        st.subheader("🌿 Predicted Growth (Sample)")
        pred_df = pd.DataFrame({
            'Actual': np.round(y_test, 2),
            'Predicted': np.round(predictions, 2)
        })
        st.dataframe(pred_df.head(10))

#=============== ENVIRONMENT MONITOR ===================
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
        st.warning("⚠️ Please upload a CSV file from the Historical Data section first.")

#================ GROWTH CONSISTENCY ====================
elif selected == "Growth Consistency":
    st.title("🌾 Growth Consistency Analysis")

    if 'df' in st.session_state:
        df = st.session_state.df
        df_numeric = df.select_dtypes(include=[np.number]).copy()

        df_numeric['plant_growth'] = (
            0.2 * df_numeric.get('pH', 0) +
            0.25 * df_numeric.get('TDS', 0) +
            0.2 * df_numeric.get('Temperature', 0) +
            0.1 * df_numeric.get('Distance (cm)', 0) +
            0.1 * df_numeric.get('LED', 0) +
            np.random.normal(0, 0.5, size=len(df_numeric))
        )

        st.subheader("📊 Environmental Stability (Std Dev)")
        env_cols = ['pH', 'TDS', 'Temperature', 'LED', 'Distance (cm)']
        available = [col for col in env_cols if col in df_numeric.columns]
        missing = [col for col in env_cols if col not in df_numeric.columns]

        if missing:
            st.warning(f"Missing columns: {', '.join(missing)}")

        stability = df_numeric[available].std().round(2)
        st.write(stability)

        st.subheader("🌿 Growth Consistency")
        growth_std = df_numeric['plant_growth'].std()
        st.metric("Growth Std Deviation", f"{growth_std:.2f}")
        st.success("✅ Stable Conditions") if growth_std <= 1.5 else st.warning("⚠️ High growth variability")

        st.subheader("📈 Environmental Trends vs. Growth")
        for col in available:
            st.line_chart(df_numeric[[col, 'plant_growth']])
    else:
        st.warning("⚠️ Please upload a CSV file from the Historical Data section first.")

#================= INSIGHTS ============================
elif selected == "Insights":
    st.title("💡 Insights & Recommendations")
    st.info("Coming soon: Smart suggestions based on plant conditions.")

#================= CONTACT =============================
elif selected == "Contact":
    st.title("📞 Contact Us")
    st.markdown("""
        **Hydro-Pi Team**  
        📧 Email: support@hydro-pi.local  
        🌍 Website: [www.hydro-pi.local](#)
    """)
