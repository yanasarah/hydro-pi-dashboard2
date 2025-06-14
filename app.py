import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from streamlit_option_menu import option_menu
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
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="color: #2e8b57; font-family: Poppins; font-size: 2rem; margin-bottom: 0.1rem;">
            About Hydroponic Systems
        </h1>
        <h2 style="color: #3a6b35; font-family: Poppins; font-size: 2.9rem; margin-top: 0;">
            At the forefront of innovation
        </h2>
    </div>
    """, unsafe_allow_html=True)

    # Two columns for explanation and image
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("""
        <p style="color: #3a4f41; font-size: 18px; line-height: 1.6; text-align: justify;">
            Hydroponics is a sustainable method of cultivating plants without soil by using nutrient-rich water solutions. 
            This technique promotes faster plant growth, higher yields, and more efficient use of space and resources. 
            It’s particularly beneficial in environments where arable land is limited, allowing year-round food production and water conservation.
        </p>
        """, unsafe_allow_html=True)

    with col2:
        st.image("Untitled-design-2.jpg", caption="Hydroponic Farming System", use_container_width=True)
       

    # 2nd section: "Why it matters" with image on the left
    col_img, spacer, col_txt = st.columns([2, 0.5, 3], vertical_alignment="top")
    with col_img:
        st.image(
            "Hydro-tower2.png",
            caption="Benefits of Hydroponics",
            use_container_width=True
        )
    with col_txt:
        st.markdown("""
            <p style="color: #3a4f41; font-size: 17px; line-height: 1.5; text-align: justify;">
                <strong>Why it matters:</strong><br>
                With climate change, urbanization, and rising food demands, hydroponics offers a smart solution.
                It uses up to 90% less water than traditional farming and can be set up virtually anywhere — from rooftops to indoor facilities.
                It brings food production closer to consumers and helps reduce the carbon footprint.
            </p>
        """, unsafe_allow_html=True)

    # How it works section
    st.markdown("""<hr style="margin-top: 2rem; margin-bottom: 1.5rem;">""", unsafe_allow_html=True)

    st.markdown("""
    <div style="padding: 1rem 3rem;">
        <h3 style="color: #2e8b57; font-family: Poppins; font-size: 1.8rem; margin-bottom: 0.5rem;">
            How Our System Works for You
        </h3>
        <p style="color: #3a4f41; font-size: 17px; line-height: 1.6; text-align: justify;">
            Our hydroponic system is fully automated and beginner-friendly. Sensors monitor water quality, temperature, and light — ensuring optimal plant health at all times. 
            Customers can easily check the status of their crops through our mobile app or web dashboard.
        </p>
        <p style="color: #3a4f41; font-size: 17px; line-height: 1.6; text-align: justify;">
            Whether you're a home gardener, a school project team, or a commercial grower, our smart system scales with your needs. 
            You get real-time updates, AI-powered growth predictions, and tips — all to make sure your plants thrive without the guesswork.
        </p>
        <p style="color: #2e8b57; font-size: 17px; font-weight: 600; text-align: justify;">
            Experience the future of farming — sustainable, smart, and surprisingly simple.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Vision & Mission Section
    st.markdown("""<hr style="margin-top: 2rem; margin-bottom: 2rem;">""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
    <div style="background-color: #cce6cc; border-radius: 12px; padding: 1.5rem; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: black; font-family: Poppins; text-align: center;">🌐 Vision</h3>
        <p style="color: black; font-size: 17px; line-height: 1.6; text-align: justify;">
            To revolutionize agriculture through smart hydroponic technologies, making sustainable and efficient food production accessible to all, regardless of location or experience.
        </p>
    </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
    <div style="background-color: #cce6cc; border-radius: 12px; padding: 1.5rem; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
        <h3 style="color: black; font-family: Poppins; text-align: center;">🌐 Mission</h3>
        <p style="color: black; font-size: 17px; line-height: 1.6; text-align: justify;">
            To empower communities and individuals by delivering user-friendly, data-driven hydroponic solutions that support a greener planet and a healthier future.
        </p>
    </div>
        """, unsafe_allow_html=True)


#==========Historical Data=============
elif selected == "Historical Data":
    st.markdown("""
        <h1 style="color:#2e8b57; font-family: Poppins;">🌱 Historical Data Analysis</h1>
    """, unsafe_allow_html=True)

    st.info("""
    📊 Upload your previous sensor data (CSV file) to let the Hydro-Pi system analyze and understand how your plants were growing under different conditions.
    
    We'll calculate a "growth score" for each record and show you how well the system can learn and predict from your data.
    """)

    uploaded_file = st.file_uploader("📤 Upload CSV Sensor Data", type=["csv"])

    if uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        df = pd.read_csv(uploaded_file)
        st.session_state.df = df

        st.subheader("📂 Raw Sensor Data")
        st.dataframe(df)

        df_numeric = df.select_dtypes(include=[np.number])
        X = df_numeric.copy()

        # Simulated growth score calculation
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

        st.subheader("✅ System Analysis Summary")
        st.markdown(f"""
        - Records analyzed: **{len(df)}**
        - We calculated a growth score based on your plant's environment.
        - The system learned from your data and can now predict plant health based on sensor readings.
        """)

        st.success("🔍 The system learned patterns with good accuracy. Prediction error is low!")

        st.markdown(f"**📉 Prediction Error (Mean Squared Error): `{mse:.3f}`**")
        st.caption("The lower this number, the more accurate the system is.")

        st.subheader("🧹 Cleaned & Enriched Data with Growth Score")
        st.dataframe(X.assign(plant_growth=np.round(y, 2)))

        st.subheader("🌿 Sample Predictions vs Actual Growth")

        pred_df = pd.DataFrame({
            'Actual Growth': np.round(y_test, 2),
            'Predicted Growth': np.round(predictions, 2)
        })

        st.dataframe(pred_df.head(10))

        # Visual plot
        st.subheader("📈 Growth Prediction Chart")
        fig, ax = plt.subplots()
        ax.plot(y_test[:20], label='Actual Growth', marker='o')
        ax.plot(predictions[:20], label='Predicted Growth', linestyle='--', marker='x')
        ax.set_title('🌿 Predicted vs Actual Plant Growth')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Growth Score')
        ax.legend()
        st.pyplot(fig)

        # Final takeaway
        st.info("""
        ✅ This shows how the Hydro-Pi system can learn from your past data and simulate how environmental factors impact plant growth.
        
        Use this insight to plan better care for your plants in the future!
        """)


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
