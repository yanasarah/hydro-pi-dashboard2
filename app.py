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

# Set Streamlit page configuration
st.set_page_config(page_title="Hydro-Pi Smart Dashboard", layout="wide")

#=============BACKGROUND , COLOUR FONT ==============
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f5fff5;
            color: #006400 !important;
            font-family: 'Poppins', sans-serif;
        }
        html, body, div, span, h1, h2, h3, h4, h5, h6, p, a, li, ul, button, label, th, td, input, textarea {
            color: #006400 !important;
            font-family: 'Poppins', sans-serif !important;
        }
        [data-testid="stSidebar"] {
            background-color: #c0ebc0 !important;
            min-width: 200px !important;
            max-width: 220px !important;
        }

        /* 🔧 Updated section below */
        [data-testid="stSidebar"] * {
            font-size: 14px !important;
            color: #003300 !important;
            background-color: transparent !important;
        }
        [data-testid="stSidebar"] > div:first-child {
            box-shadow: none !important;
            border: none !important;
            background-color: transparent !important;
        }

        .css-1dp5vir, .css-1d391kg {
            background-color: #c0ebc0 !important;
            color: #003300 !important;
            padding: 0.3rem 0.6rem !important;
            font-size: 14px !important;
            margin: 0 !important;
        }
        .stMetric label {
            color: #006400 !important;
        }
        .stDataFrame div[data-testid="stVerticalBlock"] {
            background-color: #ffffff !important;
            color: #006400 !important;
        }
        .stDataFrame thead tr th {
            background-color: #e0f5e0 !important;
            color: #006400 !important;
        }
        .stDataFrame tbody td {
            background-color: #ffffff !important;
            color: #006400 !important;
        }
        section[data-testid="stSidebar"] > div {
            width: 200px !important;
            min-width: 200px !important;
            max-width: 200px !important;
            padding-right: 0.5rem !important;
        }
        .css-1dp5vir, .css-1d391kg {
            padding: 0.3rem 0.6rem !important;
            margin: 0 !important;
            font-size: 14px !important;
        }
        section[data-testid="stSidebar"] {
            flex-shrink: 0 !important;
        }
    </style>
    """,
    unsafe_allow_html=True
)


#=================NAVIGATION==============================
with st.sidebar:
    selected = option_menu(
        menu_title="🌿 Hydro-Pi Dashboard",
        options=["Home", "About Us", "Historical Data", "Environment Monitor", "Growth Consistency", "Insights", "Contact"],
        icons=["house", "info-circle", "clock-history", "bar-chart", "activity", "lightbulb", "envelope"],
        menu_icon="cast",
        default_index=0
    )

#=====================HOME=========================
if selected == "Home":
    st.markdown("""
    <div style="padding: 2rem; background: linear-gradient(to right, #bdfcc9, #e0ffe0); border-radius: 20px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); text-align: center;">
        <h1 style="color: #2e8b57; font-family: Poppins;">🌱 Welcome to Hydro-Pi Smart Farming</h1>
        <p style="color: #4d774e; font-size: 18px;">Monitor. Predict. Grow smarter 🌿</p>
    </div>
    <br>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("""
            <div style='background-color: #e6ffe6; border-left: 5px solid #66bb66; padding: 1rem; border-radius: 12px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); font-size: 1.1rem;'>
                🌿 <em>“Grow your health, grow a garden.”</em>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div style='background-color: #ffffff; border: 2px solid #cceccc; border-radius: 12px; padding: 1rem; text-align: center; box-shadow: 0 2px 8px rgba(0,0,0,0.1);'>
                <h4 style='margin-top: 0;'>Your Plant:</h4>
                <p style='font-weight: bold; color: #2e7d32;'>🥬 Spinach</p>
                <img src="https://www.pngmart.com/files/13/Spinach-PNG-Transparent-Image.png" alt="Spinach" width="100">
            </div>
        """, unsafe_allow_html=True)

    today = datetime.now().strftime("%A, %d %B %Y")
    st.markdown(f"""
        <br>
        <div style="background-color: #e0f5e9; padding: 1.5rem; border-radius: 15px; text-align: center; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
            <h4 style="color: #1e4620;">📅 Today</h4>
            <p style="font-size: 20px; color: #1e4620;">{today}</p>
        </div>
    """, unsafe_allow_html=True)
   #===================ABOUT US===============================
elif selected == "About Us":
    st.title("🌿 About Hydro-Pi")

    st.markdown("""
    <div style="background-color: #e6ffe6; padding: 1.5rem; border-radius: 15px; box-shadow: 0 4px 10px rgba(0,0,0,0.1); font-family: 'Poppins', sans-serif;">
        <h2 style="color: #2e8b57;">What is Hydroponics?</h2>
        <p style="color: #4d774e; font-size: 16px;">
            Hydroponics is a modern method of growing plants without soil, using nutrient-rich water instead. It allows for faster growth, higher yields, and efficient use of space and resources.
        </p>
        <br>

        <h2 style="color: #2e8b57;">Why Hydroponics Matters</h2>
        <ul style="color: #4d774e; font-size: 16px;">
            <li>🌱 Grows crops faster with less water</li>
            <li>🏙️ Perfect for urban spaces and indoor farming</li>
            <li>🌍 Reduces environmental impact and pesticide use</li>
            <li>📈 Enables year-round harvest and scalable growth</li>
        </ul>
        <br>

        <h2 style="color: #2e8b57;">Introducing Hydro-Pi Smart System</h2>
        <p style="color: #4d774e; font-size: 16px;">
            Our Hydro-Pi system combines smart sensors and real-time analytics to help you monitor, analyze, and predict your plant’s health and growth. Whether you're a beginner or a commercial grower, Hydro-Pi simplifies farming decisions through:
        </p>
        <ul style="color: #4d774e; font-size: 16px;">
            <li>📊 Easy-to-understand dashboards</li>
            <li>🧠 Intelligent growth prediction using machine learning</li>
            <li>🕒 Historical trend tracking to improve yields</li>
            <li>📤 One-click raw data export for deeper insights</li>
        </ul>
        <br>

        <h2 style="color: #2e8b57;">Grow Smarter, Not Harder 🌱</h2>
        <p style="color: #4d774e; font-size: 16px;">
            With Hydro-Pi, you don’t just grow plants—you grow data-driven confidence. Join the next generation of smart farmers and make every drop of water and every ray of light count.
        </p>

        <p style="color: #2e8b57; font-size: 18px; font-weight: bold;">Hydro-Pi – Monitor. Predict. Grow smarter.</p>
    </div>
    """, unsafe_allow_html=True)


#==========Historical Data=============
elif selected == "Historical Data":
    st.markdown("""
        <h1 style="color:#2e8b57; font-family: Poppins;">Welcome to the Hydro-Pi Smart Plant System</h1>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader("📤 Upload CSV Sensor Data", type=["csv"])

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.subheader("📂 Raw Data")
        st.dataframe(df)

        df_numeric = df.select_dtypes(include=[np.number])
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
        imputer = SimpleImputer(strategy='mean')
        scaler = StandardScaler()

        X_imputed = imputer.fit_transform(X)
        X_scaled = scaler.fit_transform(X_imputed)

        y = X['plant_growth'].values
        X_features = X.drop(columns=['plant_growth'])
        X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        st.subheader("🧹 Cleaned & Enriched Data")
        st.dataframe(X.assign(plant_growth=np.round(y, 2)))

        st.subheader("📈 ML Model Performance")
        st.write(f"Mean Squared Error: {mse:.3f}")

        st.subheader("🌿 Predicted Growth (Sample)")
        pred_df = pd.DataFrame({
            'Actual': np.round(y_test, 2),
            'Predicted': np.round(predictions, 2)
        })
        st.dataframe(pred_df.head(10))

#=============ENVIRONMENT MONITOR===========================
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

#===================GROWTH CONSISTENCY=======================
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

        if growth_std > 1.5:
            st.warning("⚠️ High variability in plant growth. Consider stabilizing environmental conditions.")
        else:
            st.success("✅ Growth conditions are stable.")

        st.subheader("📈 Environmental Trends vs. Growth")
        for col in existing_env_cols:
            st.line_chart(df_numeric[[col, 'plant_growth']])
    else:
        st.warning("⚠️ Please upload a CSV file from the Historical Data section first.")

#===================INSIGHTS===============================
elif selected == "Insights":
    st.title("💡 Insights & Recommendations")
    st.info("Coming soon: Smart suggestions based on plant conditions.")

#===================CONTACT===============================
elif selected == "Contact":
    st.title("📞 Contact Us")
    st.markdown("""
        **Hydro-Pi Team**  
        📧 Email: support@hydro-pi.local  
        🌍 Website: [www.hydro-pi.local](#)
    """)
