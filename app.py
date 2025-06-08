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
#=============BACKGROUND , COLOUR FONT ==============
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

st.markdown(
    """
    <style>
        /* App background and text */
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
            max-width: 220px !important; /* ✅ NEW */
        }

        [data-testid="stSidebar"] * {
            color: #003300 !important;
        }

        .css-1dp5vir, .css-1d391kg {
            background-color: #c0ebc0 !important;
            color: #003300 !important;
            padding: 0.3rem 0.6rem !important;  /* ✅ NEW */
            font-size: 14px !important;         /* ✅ NEW */
            margin: 0 !important;               /* ✅ NEW */
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
        /* Force smaller sidebar width */
section[data-testid="stSidebar"] > div {
    width: 200px !important;
    min-width: 200px !important;
    max-width: 200px !important;
    padding-right: 0.5rem !important;
}

/* Option menu padding/fix */
.css-1dp5vir, .css-1d391kg {
    padding: 0.3rem 0.6rem !important;
    margin: 0 !important;
    font-size: 14px !important;
}

/* Ensure main content doesn't shift weirdly */
section[data-testid="stSidebar"] {
    flex-shrink: 0 !important;
}

    </style>
    """,
    unsafe_allow_html=True
)


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
from datetime import datetime

if selected == "Home":
    # Top Banner
    st.markdown("""
        <div style="
            padding: 2rem;
            background: linear-gradient(to right, #bdfcc9, #e0ffe0);
            border-radius: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            text-align: center;">
            <h1 style="color: #2e8b57; font-family: Poppins;">🌱 Welcome to Hydro-Pi Smart Farming</h1>
            <p style="color: #4d774e; font-size: 18px;">Monitor. Predict. Grow smarter 🌿</p>
        </div>
        <br>
    """, unsafe_allow_html=True)

    # Date + Upload Section Cards
    today = datetime.now().strftime("%A, %d %B %Y")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"""
            <div style="background-color: #e0f5e9; padding: 1.5rem; border-radius: 15px; text-align: center;
                        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
                <h4 style="color: #1e4620;">📅 Today</h4>
                <p style="font-size: 20px; color: #1e4620;">{today}</p>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
            <div style="background-color: #fefefe; padding: 1.5rem; border-radius: 15px; text-align: center;
                        box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
                <h4 style="color: #1e4620;">📤 Upload Sensor Data</h4>
                <p style="font-size: 16px; color: #1e4620;">Upload your CSV to view and predict plant growth trends.</p>
            </div>
        """, unsafe_allow_html=True)

    st.write("")  # spacing

    # File Upload
# File Upload
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.success("✅ File uploaded successfully!")
        st.dataframe(df)
        
    
    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

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
        st.warning("⚠️ Please upload a CSV file from the Home section first.")

#===================GROWTH CONSISTENCY=======================
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

        if growth_std > 1.5:
            st.warning("⚠️ High variability in plant growth. Consider stabilizing environmental conditions.")
        else:
            st.success("✅ Growth conditions are stable.")

        st.subheader("📈 Environmental Trends vs. Growth")
        for col in existing_env_cols:
            st.line_chart(df_numeric[[col, 'plant_growth']])
    else:
        st.warning("⚠️ Please upload a CSV file from the Home section first.")

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
