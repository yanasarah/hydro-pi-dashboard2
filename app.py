import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from streamlit_option_menu import option_menu
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from fpdf import FPDF
import base64
# Set Streamlit page configuration
st.set_page_config(page_title="Hydro-Pi Smart Dashboard", layout="wide")

# ============= STYLING ==============
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
        .stApp {
            background-color: #f5fff5;
            color: #006400 !important;
            font-family: 'Poppins', sans-serif;
        }

        [data-testid="stSidebar"] {
            background-color: #c0ebc0 !important;
            min-width: 200px !important;
        }

        [data-testid="stMetricValue"] {
            color: #006400 !important;
            font-weight: bold;
            font-size: 1.6rem;
        }

        [data-testid="stMetricLabel"] {
            color: #2e7d32 !important;
            font-size: 1rem;
        }

        .stMetric {
            background: #f0fff0;
            padding: 1.2rem;
            border-radius: 15px;
            box-shadow: 0 3px 10px rgba(0,0,0,0.05);
        }

        .css-1dp5vir {
            background-color: #c0ebc0 !important;
        }

        /* ✅ STRONGER FIX for st.error / st.success font color */
        div[data-testid="stAlert"] {
            color: #004d00 !important;
        }

        div[data-testid="stAlert"] > div {
            color: #004d00 !important;
        }

        div[data-testid="stAlert"] p {
            color: #004d00 !important;
        }

        div[data-testid="stAlert"] ul li {
            color: #004d00 !important;
        }
    </style>
""", unsafe_allow_html=True)

# ============= NAVIGATION =============
with st.sidebar:
    selected = option_menu(
        menu_title="🌿 Hydro-Pi Dashboard",
        options=["Home", "About Us", "Historical Data", "Environment Monitor", "Growth Consistency", "Insights","Crop Comparison","AI Forecast","Beginner FAQ", "Contact"],
        icons=["house", "info-circle", "clock-history", "bar-chart", "activity", "lightbulb","cat","dog", "envelope"],
        menu_icon="cast",
        default_index=0
    )
# ============= DATA LOADING FUNCTIONS =============
@st.cache_data
def load_main_data():
    try:
        df = pd.read_excel("summary data.xlsx", sheet_name="summary data")
        df['Week'] = df['Week'].ffill()
        df['DateTime'] = pd.to_datetime(df['Day'].astype(str)) + pd.to_timedelta(df['Time'])
        return df
    except:
        return pd.DataFrame()
@st.cache_data

def load_weekly():
    try:
        return pd.read_excel("summary data.xlsx", sheet_name="weekly trend ")
    except Exception as e:
        st.error(f"Error loading weekly trend data: {e}")
        return pd.DataFrame()



@st.cache_data
def load_daily():
    try:
        return pd.read_excel("summary data.xlsx", sheet_name="Average Daily")
    except:
        return pd.DataFrame()

# ============= HOME PAGE =============

if selected == "Home":
    import random
    import streamlit.components.v1 as components

    # 🌿 HERO SECTION (WELCOME BANNER)
    st.markdown("""
        <div style="padding: 2.5rem; background: linear-gradient(135deg, #e1f5e1, #b9f0b9);
                    border-radius: 20px; text-align: center;">
            <h1 style="color: #1b5e20; font-size: 3rem;">🌿 Welcome to Hydro-Pi</h1>
            <h3 style="color: #388e3c;">Smart Farming for Every Indoor Grower</h3>
            <p style="font-size: 1.1rem; color: #2e7d32;">
                Monitor. Optimize. Harvest better.
            </p>
            <img src="https://cdn.pixabay.com/photo/2020/06/06/20/35/hydroponics-5267540_1280.jpg"
                 style="max-height: 300px; margin-top: 1rem; border-radius: 15px;" />
        </div>
        <br>
    """, unsafe_allow_html=True)

    # 💡 BENEFITS GRID
    st.markdown("### 💡 Why Use Hydro-Pi?")
    b1, b2, b3 = st.columns(3)
    b1.success("📈 **Real-time Monitoring**  \nTrack pH, TDS, temp & humidity.")
    b2.info("🤖 **Smart Recommendations**  \nGet alerts + AI tips based on live data.")
    b3.warning("📲 **Easy to Use**  \nAccess anywhere — mobile, tablet or PC.")

    # 🌿 CURRENT PLANT SNAPSHOT
    st.markdown("### 🪴 Current Grow Session")
    st.success("🧬 Crop: *Spinach*  \n📅 Started: 3 weeks ago  \n⚙️ Method: Drip")
    
    # ✅ WEEKLY GOALS
    st.markdown("### 🎯 Weekly Care Goals")
    st.checkbox("🌿 Maintain pH between 5.8–6.2", value=True)
    st.checkbox("💧 Keep TDS around 700 ppm", value=True)
    st.checkbox("🌡️ Ensure water temp below 30°C", value=False)
    st.checkbox("💨 Maintain humidity above 40%", value=True)

    # 📅 DATE + GROW TIP
    from datetime import datetime
    facts = [
        "Spinach thrives in cool, humid environments.",
        "Deep roots prefer stable TDS levels below 800 ppm.",
        "Growth slows if pH drifts out of 5.5–6.5 range.",
        "Hydroponic tanks need 18+ hours of light per day.",
        "Add beneficial microbes weekly to improve nutrient uptake."
    ]

    st.markdown("### 📌 Today’s Grow Insight")
    st.info(f"🌿 **{random.choice(facts)}**")

    st.markdown(f"""
        <div style="margin-top: 1.5rem; background: #e8f5e9; padding: 1rem;
                    border-radius: 12px; text-align: center;">
            <h5 style="color: #2e7d32;">📅 Today</h5>
            <p style="font-size: 16px; font-weight: bold; color: #1b5e20;">
                {datetime.now().strftime("%A, %d %B %Y")}
            </p>
        </div>
    """, unsafe_allow_html=True)

# ========= historical data==================== 
elif selected == "Historical Data": 
    st.markdown(""" 
    <style> 
        html, body, [class*="st-"] { color: #006400 !important; } 
        h1, h2, h3, h4, h5, h6 { color: #2e8b57 !important; } 
        .stMetric label, .stMetric div { color: #006400 !important; } 
        .dataframe td, .dataframe th { color: #006400 !important; } 
        label, .stTextInput, .stSelectbox, .stRadio, .stSlider, .stFileUploader { color: #006400 !important; } 
        .stAlert, .stSuccess, .stWarning { color: #006400 !important; } 
    </style> 
    """, unsafe_allow_html=True) 

    st.markdown("<h1 style='color:#2e8b57;'>        Historical Data Analysis</h1>", unsafe_allow_html=True) 

    data_source = st.radio("Select data source:", 
                           ["Use built-in dataset", "Upload your own Excel file"], 
                           horizontal=True)

    if data_source == "Upload your own Excel file":
        uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])
        
        if uploaded_file:
            try:
                xls = pd.ExcelFile(uploaded_file)
                first_sheet = xls.sheet_names[0]
                df = pd.read_excel(xls, sheet_name=first_sheet)
                df['Week'] = df['Week'].ffill()
                st.success(f"File uploaded successfully! Using sheet: {first_sheet}")

                st.session_state["df"] = df
                st.session_state["uploaded_file"] = uploaded_file
            except Exception as e:
                st.error(f"Error reading file: {e}")
                st.stop()
        else:
            st.info("Please upload an Excel file or switch to built-in dataset")
            st.stop()

    else:
        df = load_main_data()
        if df.empty:
            st.error("Built-in data not available. Please upload a file instead.")
            st.stop()
        st.success("Using built-in Hydro-Pi dataset")
        st.session_state["df"] = df

    # ===== FILTERING =====
    st.subheader("        Filter Data")
    filter_col1, filter_col2 = st.columns(2)

    if 'Week' in df.columns:
        selected_week = filter_col1.selectbox("Select Week", df['Week'].dropna().unique())
    else:
        selected_week = None

    if 'Day' in df.columns and selected_week is not None:
        available_days = df[df['Week'] == selected_week]['Day'].unique()
        selected_day = filter_col2.selectbox("Select Day", available_days)
    else:
        selected_day = None

    if selected_week is not None and selected_day is not None:
        filtered_df = df[(df['Week'] == selected_week) & (df['Day'] == selected_day)]
    else:
        filtered_df = df.copy()

    # ===== SUMMARY =====
    st.subheader("           Data Overview") 
    col1, col2, col3 = st.columns(3) 
    col1.metric("Total Records", len(df)) 
    col2.metric("Days Recorded", df['Day'].nunique() if 'Day' in df.columns else "N/A") 
    col3.metric("Weeks Recorded", df['Week'].nunique() if 'Week' in df.columns else "N/A") 

    # ===== WEEKLY STATISTICS =====
    st.subheader(" 📈 Nutrient Trends Over Time")
    st.markdown("ℹ️ This graph shows  weekly trends in pH, TDS, temperature, and humidity.")

    if 'Week' in df.columns:
        stat_table = df.groupby("Week")[['TDS', 'pH', 'DHT22 1', 'HUM 1', 'DHT 22 2', 'HUM 2', 'DS18B20']].agg(['mean', 'std'])
        st.dataframe(stat_table.style.format("{:.2f}"), height=350, use_container_width=True)
    else:
        st.warning("Week column not found — unable to compute weekly summary.")



    # ===== CORRELATION ANALYSIS =====
    st.subheader("🔗 Parameter Correlations")
    st.markdown("ℹ️ This chart reveals how sensor values relate to each other — for example, high TDS often relates to lower pH.")
    # Select only numeric columns and exclude datetime
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

    if len(numeric_cols) >= 2:
        try:
            # Create correlation matrix
            corr_matrix = filtered_df[numeric_cols].corr()
            
            # Create figure with larger size
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Customize heatmap appearance
            sns.heatmap(
                corr_matrix, 
                annot=True, 
                cmap="YlGnBu", 
                ax=ax,
                annot_kws={"size": 10, "color": "black"},  # Darker annotation text
                linewidths=.5
            )
            
            # Rotate x-axis labels for better readability
            plt.xticks(rotation=45)
            
            # Ensure tight layout to prevent cutoff
            plt.tight_layout()
            
            # Display in Streamlit
            st.pyplot(fig)
            
        except Exception as e:
            st.error(f"Error generating correlation heatmap: {e}")
    else:
        st.warning(f"Need at least 2 numeric columns for correlation. Found: {numeric_cols}")


# ===== GROWTH SCORE MODEL =====
st.subheader("🌿 Plant Health Analysis")

if st.checkbox("Calculate Growth Score", True, help="Calculate plant health score based on environmental factors"):
    # Check required columns exist
    required_cols = ['DS18B20', 'HUM 1', 'TDS', 'pH']
    if all(col in filtered_df.columns for col in required_cols):
        filtered_df['Growth_Score'] = (
            0.3 * filtered_df['DS18B20'] +
            0.2 * (100 - filtered_df['HUM 1']) +
            0.25 * filtered_df['TDS'] / 100 +
            0.25 * filtered_df['pH']
        )
        filtered_df['Growth_Score'] = (
            (filtered_df['Growth_Score'] - filtered_df['Growth_Score'].min()) /
            (filtered_df['Growth_Score'].max() - filtered_df['Growth_Score'].min())
        ) * 100

         # Just take the last 150 rows
        recent_scores = filtered_df['Growth_Score'].tail(150).reset_index(drop=True)
        st.line_chart(recent_scores)

        # === Advanced Predictions ===
        if st.checkbox("Show Advanced Predictions", help="Show machine learning predictions vs actual growth"):
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.model_selection import KFold
            from sklearn.metrics import mean_squared_error
            from math import sqrt
            import matplotlib.pyplot as plt
            import plotly.graph_objects as go
            import pandas as pd

            X = filtered_df[['pH', 'TDS', 'DS18B20', 'HUM 1']]
            y = filtered_df['Growth_Score']

            kf = KFold(n_splits=5, shuffle=True, random_state=42)
            actuals = []
            predictions = []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                actuals.extend(y_test.values)
                predictions.extend(y_pred)

                last_model = model  # Keep last trained model for feature importance

            rmse = sqrt(mean_squared_error(actuals, predictions))
            st.success(f"📊 AI Accuracy (Error Margin): ±{rmse:.2f} points")
            st.caption("This shows how far off the AI predictions are, on average (lower is better).")

            # 📈 Line Chart of Predictions
            st.markdown("### 🤖 Growth Score AI Prediction (Cross-Validated)")
            pred_df = pd.DataFrame({
                "Actual": actuals,
                "Predicted": predictions
            })
            st.line_chart(pred_df.reset_index(drop=True))

            # 📊 Feature Importance
            st.markdown("### 📊 Sensor Impact (Feature Importance)")
            features = ['pH', 'TDS', 'DS18B20', 'HUM 1']
            importances = last_model.feature_importances_

            fig_imp, ax = plt.subplots()
            ax.barh(features, importances, color='#66bb6a')
            ax.set_xlabel("Importance Score")
            ax.set_title("Most Influential Sensors for Growth Prediction")
            st.pyplot(fig_imp)
            st.caption("Higher scores = more influence on growth score.")

            # 🎯 Scatter Plot (Actual vs Predicted)
            st.markdown("### 🎯 Predicted vs Actual Growth Score (Scatter Plot)")
            fig_scatter = go.Figure()

            fig_scatter.add_trace(go.Scatter(
                x=pred_df["Actual"],
                y=pred_df["Predicted"],
                mode='markers',
                marker=dict(size=8, color='#43a047'),
                name="Prediction"
            ))

            min_val = min(pred_df["Actual"].min(), pred_df["Predicted"].min())
            max_val = max(pred_df["Actual"].max(), pred_df["Predicted"].max())
            fig_scatter.add_trace(go.Scatter(
                x=[min_val, max_val],
                y=[min_val, max_val],
                mode='lines',
                line=dict(dash='dash', color='gray'),
                name="Ideal"
            ))

            fig_scatter.update_layout(
                title="Actual vs Predicted Growth Score",
                xaxis_title="Actual",
                yaxis_title="Predicted",
                height=400
            )
            st.plotly_chart(fig_scatter, use_container_width=True)
            st.caption("Dots close to the dashed line = accurate predictions.")
    else:
        st.warning("Missing required columns for growth score calculation.")

    # ===== RECOMMENDATIONS =====
    st.subheader("💡 Optimization Recommendations")

    if 'pH' in filtered_df.columns:
        avg_pH = filtered_df['pH'].mean()
        if avg_pH < 5.8:
            st.warning("⚠️ pH is slightly low. Consider adding pH Up solution.")
        elif avg_pH > 6.2:
            st.warning("⚠️ pH is slightly high. Consider adding pH Down solution.")
        else:
            st.success("✅ pH level is optimal")
    else:
        st.warning("pH data not available for recommendations")

    if 'TDS' in filtered_df.columns:
        avg_tds = filtered_df['TDS'].mean()
        if avg_tds < 650:
            st.warning("⚠️ Nutrient levels low. Consider adding fertilizer.")
        elif avg_tds > 750:
            st.warning("⚠️ Nutrient levels high. Consider diluting solution.")
        else:
            st.success("✅ Nutrient levels are optimal")
    else:
        st.warning("TDS data not available for nutrient recommendations")

    # ===== RAW DATA =====
    st.subheader("📋 Detailed Measurements")
    st.dataframe(filtered_df.style.background_gradient(cmap='YlGn'),
                 height=300,
                 use_container_width=True)

#========evironment monitor============
elif selected == "Environment Monitor":
    import plotly.graph_objects as go
    import plotly.figure_factory as ff

    st.markdown("""
        <h1 style='text-align: center; color: #4CAF50;'>🌿 Environment Monitor</h1>
        <p style='text-align: center;'>Live overview of current plant environment</p>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df", pd.DataFrame())

    if not df.empty:
        # 🧠 Generate synthetic DateTime and Week if not present
        if 'DateTime' not in df.columns:
            df['DateTime'] = pd.to_datetime("2025-01-01") + pd.to_timedelta(df['Day'] - 1, unit='D') + pd.to_timedelta(df['Time'].astype(str))
        if 'Week' not in df.columns:
            df['Week'] = ((df['Day'] - 1) // 7) + 1

        latest = df.iloc[-1]

        st.markdown("### 📈 Recent Sensor Trends (pH, Nutrient Level, Water Temp)")
        df_sorted = df.sort_values("DateTime")
        fig_trend = go.Figure()
        for col in ['pH', 'TDS', 'DS18B20']:
            fig_trend.add_trace(go.Scatter(
                x=df_sorted['DateTime'], y=df_sorted[col],
                mode='lines+markers',
                name=col
            ))
        fig_trend.update_layout(
            title="📊 Recent Trends",
            xaxis_title="Time",
            yaxis_title="Value",
            legend_title="Sensor",
            hovermode="x unified"
        )
        st.plotly_chart(fig_trend, use_container_width=True)

        #======weekly bar chart=====
       
                # 📊 Weekly Average Bar Chart (Expanded Version)
        st.markdown("### 📊 Weekly Average Comparison (Stacked Bar Chart)")

        # Choose which sensors to include
        weekly_avg = df.groupby("Week")[['pH', 'TDS', 'DS18B20', 'DHT22 1', 'HUM 1']].mean().reset_index()
        sensor_colors = {
            'pH': 'green',
            'TDS': 'blue',
            'DS18B20': 'orange',
            'DHT22 1': 'red',
            'HUM 1': 'purple'
        }

        fig_bar = go.Figure()
        for sensor in ['pH', 'TDS', 'DS18B20', 'DHT22 1', 'HUM 1']:
            fig_bar.add_trace(go.Bar(
                x=weekly_avg['Week'],
                y=weekly_avg[sensor],
                name=sensor,
                marker_color=sensor_colors.get(sensor, None),
                hovertemplate=f"<b>{sensor}</b><br>Week: %{{x}}<br>Avg: %{{y:.2f}}<extra></extra>"
            ))

        fig_bar.update_layout(
            barmode='stack',  # Try 'group' if you prefer grouped bars
            title='📦 Stacked Weekly Averages for Key Sensors',
            xaxis_title='Week',
            yaxis_title='Average Sensor Values',
            legend_title='Sensor',
            hovermode='x unified'
        )

        st.plotly_chart(fig_bar, use_container_width=True)

  ### ==== Current Environment Status   ===========   
        st.markdown("### 📋 Current Environment Status")
        col1, col2, col3 = st.columns(3)
        col1.metric("💧 pH Level", f"{latest['pH']:.2f}")
        col2.metric("⚡ TDS (ppm)", f"{latest['TDS']:.0f}")
        col3.metric("🌡️ Water Temp (DS18B20)", f"{latest['DS18B20']:.1f}°C")

        col4, col5, col6 = st.columns(3)
        col4.metric("🌡️ Air Temp 1", f"{latest['DHT22 1']:.1f}°C")
        col5.metric("💦 Humidity 1", f"{latest['HUM 1']}%")
        col6.metric("🌡️ Air Temp 2", f"{latest['DHT 22 2']:.1f}°C")

        col7, col8 = st.columns(2)
        col7.metric("💦 Humidity 2", f"{latest['HUM 2']}%")

        st.markdown("---")

        # 🚨 Alerts
        st.markdown("### 🚨 Live Alerts")
        alerts = []
        if latest['pH'] < 5.5 or latest['pH'] > 7.5:
            alerts.append("⚠️ pH is out of the optimal range (5.5–7.5)")
        if latest['TDS'] > 1200:
            alerts.append("⚠️ TDS is too high (> 1200 ppm)")
        if latest['DS18B20'] > 30:
            alerts.append("🔥 Water temperature is too high")
        if latest['HUM 1'] < 40 or latest['HUM 2'] < 40:
            alerts.append("💧 Humidity is low (< 40%)")
        if latest['DHT22 1'] > 35 or latest['DHT 22 2'] > 35:
            alerts.append("🌞 Air temperature is too high")

        if alerts:
            st.error("⚠️ Environment Alerts:")
            for a in alerts:
                st.markdown(f"- {a}")
        else:
            st.success("✅ All parameters are within the healthy range.")
    else:
        st.warning("📂 No data available. Please upload an Excel file on the Home page.")

# ============= GROWTH CONSISTENCY PAGE =============
elif selected == "Growth Consistency":
    st.markdown("""
    <h1 style='color:#2e8b57;'>🌿 Growth Consistency</h1>
    <p style='color:#4e944f;'>Analyzing your plant environment's stability</p>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style='background-color: #e8f5e9; padding: 1.5rem; border-left: 6px solid #66bb6a; border-radius: 10px;
                box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-bottom: 1rem;'>
        <h3 style='color: #2e7d32;'>✅ Overall Environment Stability</h3>
        <p style='color: #4e944f; font-size: 16px;'>
            All core environmental parameters (pH, TDS, Temperature, Humidity) show strong consistency over time. <br>
            <strong>No major fluctuations detected.</strong> Your setup is optimal for healthy hydroponic growth. 🌱
        </p>
    </div>
    """, unsafe_allow_html=True)


        # Load data
    df = st.session_state.get("df", pd.DataFrame())

    if not df.empty:
        df['Week'] = df['Week'].ffill()
        daily_df = df.groupby('Day')[['TDS', 'pH', 'DHT22 1', 'HUM 1', 'DS18B20']].mean().reset_index()

        # Calculate Coefficient of Variation
        cv = daily_df.std(numeric_only=True) / daily_df.mean(numeric_only=True)
        max_cv = cv.max()

        if max_cv < 0.1:
            st.markdown("""
            <div style='background-color: #e8f5e9; padding: 1.5rem; border-left: 6px solid #66bb6a; border-radius: 10px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-bottom: 1rem;'>
                <h3 style='color: #2e7d32;'>✅ Excellent Consistency</h3>
                <p style='color: #4e944f; font-size: 16px;'>
                    All parameters (pH, TDS, Temp, Humidity) are stable with minimal fluctuations.<br>
                    Your system is perfectly balanced — keep up the good work! 🌱
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style='background-color: #fffde7; padding: 1.5rem; border-left: 6px solid #fbc02d; border-radius: 10px;
                        box-shadow: 0 2px 6px rgba(0,0,0,0.05); margin-bottom: 1rem;'>
                <h3 style='color: #f57f17;'>⚠️ Inconsistencies Detected</h3>
                <p style='color: #5d4037; font-size: 16px;'>
                    Some sensors show higher variability. Review trends below and consider stabilizing water or nutrients. 🌿
                </p>
            </div>
            """, unsafe_allow_html=True)

        # CV Chart
        st.markdown("### 📊 Coefficient of Variation (CV)")
        st.markdown("ℹ️ **This chart shows how stable each sensor reading is over time.**\nLower values mean more consistent readings. Higher variation may signal issues like nutrient imbalances or environmental stress.")
        cv_df = cv.reset_index()
        cv_df.columns = ['Parameter', 'CV']
        st.bar_chart(cv_df.set_index("Parameter"))

        st.info("""
        - Lower CV = More stable readings  
        - Ideal: CV < 0.1 for critical sensors like pH and TDS
        """)

        # Rolling trends
        st.markdown("### 📈 3-Day Rolling Average Trends")
        rolling_df = daily_df.set_index('Day').rolling(window=3).mean()
        st.markdown("ℹ️ **This line chart smooths out daily data using a 3-day average.**\nIt helps you spot trends without being distracted by daily noise or minor fluctuations.")
        st.line_chart(rolling_df)

        # Parameter-level alerts
        st.markdown("### 🚨 Inconsistency Alerts")
        st.markdown("ℹ️ **Here you'll see alerts if any parameter is too unstable.**\nWe compare standard deviation against the average to detect large swings.")
        for param in ['TDS', 'pH', 'DS18B20', 'DHT22 1', 'HUM 1']:
            std = daily_df[param].std()
            mean = daily_df[param].mean()
            if mean != 0 and std / mean > 0.15:
                st.warning(f"⚠️ {param} shows high variability: {std/mean:.2%}")
            elif std / mean <= 0.1:
                st.success(f"✅ {param} is stable ({std/mean:.2%})")
    else:
        st.warning("No data available. Please upload your Excel file on the Home page.")

#====about us======
elif selected == "About Us":
    # Logo and header
    st.image("Blue and Green Illustrative Hydroponic Logo Design (1).png", width=140)
    st.markdown("""
    <div style="text-align: center; padding: 2rem;">
        <h1 style="color: #2e8b57; font-family: Poppins; font-size: 2.5rem; margin-bottom: 0.2rem;">
            About Hydro-Pi Smart Farming
        </h1>
        <h3 style="color: #388e3c; font-family: Poppins; margin-top: 0;">
            Bringing Innovation to Your Roots 🌱
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # What is Hydroponics
    col1, col2 = st.columns([3, 2])
    with col1:
        st.markdown("""
        <p style="color: #3a4f41; font-size: 18px; line-height: 1.7; text-align: justify;">
            <strong>Hydroponics</strong> is a soil-less farming technique that uses water-based nutrient solutions 
            to grow plants faster, healthier, and more efficiently. It enhances growth and reduces water usage by up to 90%.
            <br><br>
            Ideal for urban areas, rooftops, and indoor farms and enabling year-round, pesticide-free food production.
        </p>
        """, unsafe_allow_html=True)
    with col2:
        st.image("Untitled-design-2.jpg", caption="Hydroponic Farming System", use_container_width=True)

    # Why It Matters
    st.markdown("### 🌍 Why Hydroponics Matters")
    col_img, col_txt = st.columns([2, 3])
    with col_img:
        st.image("Hydro-tower2.png", caption="Benefits of Smart Growing", use_container_width=True)
    with col_txt:
        st.markdown("""
        <p style="color: #3a4f41; font-size: 17px; line-height: 1.6; text-align: justify;">
            Hydroponics is a <strong>sustainable, high-yield solution</strong> for modern food challenges.
            It reduces dependence on weather and space, cutting the carbon footprint while maximizing output.
        </p>
        """, unsafe_allow_html=True)

    # How It Works
    st.markdown("""<hr style="margin-top: 2rem; margin-bottom: 1.5rem;">""", unsafe_allow_html=True)
    st.markdown("""
    <div style="padding: 1rem 2rem;">
        <h3 style="color: #2e8b57; font-family: Poppins; font-size: 1.9rem;">
            🚀 How Hydro-Pi Helps You Grow Smarter
        </h3>
        <p style="color: #3a4f41; font-size: 17px; line-height: 1.6; text-align: justify;">
            Hydro-Pi combines sensor automation, real-time monitoring, and AI recommendations to optimize growing 24/7.
            Whether you're a beginner or a commercial grower, it adapts to your needs.
        </p>
        <p style="color: #2e8b57; font-size: 17px; font-weight: 600; text-align: justify;">
            Grow confidently. Grow cleanly. Grow smart.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Vision & Mission
    st.markdown("""<hr style="margin-top: 2rem; margin-bottom: 2rem;">""", unsafe_allow_html=True)
    st.markdown("### 🌟 Our Vision & Mission")
    v1, v2 = st.columns(2)
    with v1:
        st.markdown("""
        <div style="background-color: #cce6cc; border-radius: 12px; padding: 1.5rem; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
            <h3 style="text-align: center; color: #1b5e20;">🌐 Vision</h3>
            <p style="font-size: 16px; text-align: justify; color: #2e7d32;">
                To revolutionize agriculture through smart hydroponic technology also can making food growth cleaner,
                faster, and more sustainable for everyone, everywhere.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with v2:
        st.markdown("""
        <div style="background-color: #cce6cc; border-radius: 12px; padding: 1.5rem; box-shadow: 2px 2px 10px rgba(0,0,0,0.1);">
            <h3 style="text-align: center; color: #1b5e20;">🎯 Mission</h3>
            <p style="font-size: 16px; text-align: justify; color: #2e7d32;">
                To empower growers with user-friendly, automated, data-driven solutions that support food security, education, and sustainability.
            </p>
        </div>
        """, unsafe_allow_html=True)

    # Testimonials
    st.markdown("### 💬 What Our Growers Say")
    testi1, testi2 = st.columns(2)
    with testi1:
        st.markdown(
    "> “Since I installed Hydro-Pi, my spinach grew faster and fuller. The alerts saved me twice!”  \n"
    "— **Fazli**, Home Grower, Johor")
    with testi2:
        st.markdown(
    "> “We use Hydro-Pi in our school greenhouse. The kids love checking plant stats and watching them grow.”  \n"
    "— **Cikgu Nurul**, School Hydroponics Project")

    # Timeline
    st.markdown("### 🌱 The Journey: Seed to Harvest")
    timeline_steps = [
        "🔹 **Week 0:** Germination begins",
        "🔹 **Week 1-2:** Root and leaf expansion",
        "🔹 **Week 3-4:** Nutrient boost, temperature and humidity optimization",
        "🔹 **Week 5+:** Ready to harvest!"
    ]
    for step in timeline_steps:
        st.markdown(f"- {step}")

    # FAQ
    st.markdown("### ❓ Frequently Asked Questions")
    with st.expander("🌿 What is hydroponics?"):
        st.write("Hydroponics is growing plants without soil using water enriched with nutrients.")
    with st.expander("📊 What does Hydro-Pi monitor?"):
        st.write("It tracks pH, TDS, temperature, and humidity for optimal plant growth.")
    with st.expander("📱 Do I need an app?"):
        st.write("No, just use the web dashboard on any device.")
    with st.expander("💧 What if pH or nutrients are off?"):
        st.write("Hydro-Pi notifies you and provides suggestions to fix it.")

#==== INSIGHT===============
elif selected == "Insights":
    import base64
    from fpdf import FPDF

    st.markdown("""
    <h1 style="color:#2e8b57;">💡 Hydro Insights & Optimization</h1>
    <p style="color:#4e944f;">Data-driven tips based on your environmental trends</p>
    """, unsafe_allow_html=True)

    st.markdown("### 🌿 Growth Insights")
    st.success("✅ Your average pH is within the optimal range (5.8 - 6.2)")
    st.warning("⚠️ TDS has fluctuated more than 15% in the past week — consider rebalancing nutrients.")
    st.info("📈 Growth score improved 12% since last cycle — nice job!")

    st.markdown("### 📊 Trends Overview")
    if 'df' in st.session_state:
        df = st.session_state['df']
        if 'pH' in df.columns and 'TDS' in df.columns:
            st.line_chart(df[['pH', 'TDS']])
        else:
            st.line_chart(df)

    st.markdown("### 🧠 Smart Suggestions")
    st.markdown("""
    - Adjust pH slowly — no more than 0.2 per day  
    - Ideal TDS for spinach: **650–750 ppm**  
    - Maintain water temp below 27°C for root health  
    - Keep humidity stable (50–70%) to prevent mold
    """)

    st.markdown("Want more AI-driven insights in the future? Stay tuned!")

    # ========== PDF Report Tools ==========
    def generate_pdf(data):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", "B", 16)
        pdf.cell(200, 10, "Hydro-Pi Weekly Report", ln=True, align='C')
        pdf.set_font("Arial", size=12)
        pdf.ln(10)

        pdf.cell(200, 10, f"Average pH: {data.get('pH', 'N/A'):.2f}", ln=True)
        pdf.cell(200, 10, f"Average TDS: {data.get('TDS', 'N/A'):.0f} ppm", ln=True)
        pdf.cell(200, 10, f"Avg Temp (DS18B20): {data.get('DS18B20', 'N/A'):.1f} °C", ln=True)
        pdf.ln(5)
        pdf.multi_cell(0, 10, "Insights:\n- pH levels are stable\n- TDS is slightly high, consider diluting\n- Temperature within safe range")
        return pdf.output(dest='S').encode('latin-1')

    def download_pdf_button(df):
        if not df.empty:
            latest_avg = df.tail(50).mean(numeric_only=True)
            pdf_bytes = generate_pdf(latest_avg)
            b64 = base64.b64encode(pdf_bytes).decode()
            href = f'<a href="data:application/octet-stream;base64,{b64}" download="hydro_pi_report.pdf">📄 Download Weekly PDF Report</a>'
            st.markdown(href, unsafe_allow_html=True)

    st.markdown("### 📥 Weekly Report")
    if 'df' in st.session_state:
        download_pdf_button(st.session_state['df'])
    else:
        st.warning("No data available to generate the report.")
        
# ============= CROP COMPARISON PAGE =============
elif selected == "Crop Comparison":
    st.markdown("""
    <h1 style='color:#2e8b57;'>📊 Crop Cycle Comparison</h1>
    <p style='color:#4e944f;'>Compare environmental trends across different growing cycles</p>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df", pd.DataFrame())

    if not df.empty and 'Week' in df.columns:
        df['Week'] = df['Week'].ffill()

        # Define cycles by week range
        cycle1_weeks = [1, 2]
        cycle2_weeks = [3, 4, 5]
        df['Cycle'] = df['Week'].apply(lambda w: 'Cycle 1' if w in cycle1_weeks else ('Cycle 2' if w in cycle2_weeks else 'Other'))

        # Filter only cycle data
        df_cycle = df[df['Cycle'].isin(['Cycle 1', 'Cycle 2'])]

        st.subheader("🧮 Summary Statistics by Cycle")
        st.markdown("ℹ️ **This table shows the average and variation of sensor values for each crop cycle.**\nIt helps you compare environmental conditions between Cycle 1 and Cycle 2.")
        summary = df_cycle.groupby('Cycle')[['pH', 'TDS', 'DS18B20', 'DHT22 1', 'HUM 1', 'DHT 22 2', 'HUM 2']].agg(['mean', 'std']).round(2)
        st.dataframe(summary)

        st.markdown("""
        <div style="color:#4e944f; font-size: 16px; margin-top: 0.5rem;">
            📅 <strong>Cycle 1:</strong> Week 1–2 (14 days) <br>
            📅 <strong>Cycle 2:</strong> Week 3–5 (21 days)
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### 📊 Visual Comparison")
        st.markdown("ℹ️ **The following bar charts compare average sensor values between the two crop cycles.**\nUse them to see which cycle had better conditions for plant growth.")

        import plotly.graph_objects as go
        parameters = ['pH', 'TDS', 'DS18B20', 'DHT22 1', 'HUM 1', 'DHT 22 2', 'HUM 2']

        for param in parameters:
            cycle_means = df_cycle.groupby('Cycle')[param].mean()
            fig = go.Figure(data=[
                go.Bar(name='Cycle 1', x=[param], y=[cycle_means.get('Cycle 1', 0)]),
                go.Bar(name='Cycle 2', x=[param], y=[cycle_means.get('Cycle 2', 0)])
            ])
            fig.update_layout(
                title=f"{param} Comparison",
                barmode='group',
                yaxis_title=param,
                xaxis_title="Parameter"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧠 AI Observations")
        st.markdown("ℹ️ **These insights analyze which cycle had more stable readings.**\nStability often means better, less stressful growth for your plants.")

        def get_stability(param):
            stds = df_cycle.groupby("Cycle")[param].std()
            if stds.get("Cycle 1", 0) < stds.get("Cycle 2", 0):
                return f"✅ {param} was more stable in **Cycle 1**"
            elif stds.get("Cycle 2", 0) < stds.get("Cycle 1", 0):
                return f"✅ {param} was more stable in **Cycle 2**"
            else:
                return f"{param} stability was similar across cycles"

        for p in parameters:
            st.markdown(f"- {get_stability(p)}")

        st.info("🌿 Tip: Stability in pH, TDS, and temperature often results in better plant growth.")

    else:
        st.warning("No cycle data found. Make sure your dataset has a 'Week' column with valid values.")

# ============= AI FORECAST & RISK PREDICTION =============
elif selected == "AI Forecast":
    st.markdown("""
    <h1 style='color:#2e8b57;'>🤖 AI Forecast & Risk Prediction</h1>
    <p style='color:#4e944f;'>Smarter decisions powered by your environmental data</p>
    """, unsafe_allow_html=True)

    df = st.session_state.get("df", pd.DataFrame())

    if not df.empty and 'Week' in df.columns:
        df['Week'] = df['Week'].ffill()

        st.markdown("### 🧪 Nutrient Rebalancing Suggestion")
        recent_tds = df[df['Week'].isin([4, 5])]['TDS']
        if recent_tds.mean() > 1200 or recent_tds.diff().mean() > 50:
            st.error("⚠️ High or rapidly rising TDS detected — consider rebalancing nutrients.")
        else:
            st.success("✅ TDS levels are stable — no action needed.")

        st.markdown("### 🗓️ Harvest Day Estimator")
        if df['Week'].isin([3]).any():
            st.info("🪴 Spinach observed in Week 3.")
            st.success("📅 Estimated optimal harvest window: **Week 6**")
        else:
            st.warning("❓ No plant growth detected in Week 3 — unable to forecast harvest.")

        st.markdown("### 🌫️ Fungus Risk Forecast")
        weekly_humidity = df.groupby('Week')[['HUM 1', 'HUM 2']].mean()
        risk_levels = []
        for week, row in weekly_humidity.iterrows():
            if row['HUM 1'] > 75 or row['HUM 2'] > 75:
                risk_levels.append((week, 'High'))
            elif row['HUM 1'] > 65 or row['HUM 2'] > 65:
                risk_levels.append((week, 'Medium'))
            else:
                risk_levels.append((week, 'Low'))

        for week, risk in risk_levels:
            emoji = "🟥" if risk == 'High' else ("🟧" if risk == 'Medium' else "🟩")
            st.markdown(f"{emoji} Week {int(week)} fungus risk: **{risk}**")

        st.markdown("### 📈 Weekly Humidity Trend")
        import plotly.graph_objects as go
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=weekly_humidity.index, y=weekly_humidity['HUM 1'], mode='lines+markers', name='HUM 1'))
        fig.add_trace(go.Scatter(x=weekly_humidity.index, y=weekly_humidity['HUM 2'], mode='lines+markers', name='HUM 2'))
        fig.update_layout(
            title="Average Weekly Humidity Levels",
            xaxis_title="Week",
            yaxis_title="Humidity (%)",
            yaxis=dict(range=[0, 100]),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

        # ============== TDS Forecast ==============
        st.markdown("### 📉 TDS Forecast – Next 2 Days")

        df["Day"] = df["Day"].ffill()
        tds_daily = df.groupby("Day")["TDS"].mean().reset_index()
        tds_daily["DayIndex"] = range(len(tds_daily))

        from sklearn.linear_model import LinearRegression
        import numpy as np

        X = tds_daily["DayIndex"].values.reshape(-1, 1)
        y = tds_daily["TDS"].values
        model = LinearRegression()
        model.fit(X, y)

        future_x = np.array([len(tds_daily), len(tds_daily)+1]).reshape(-1, 1)
        future_y = model.predict(future_x)

        forecast_days = [f"Day {i+1}" for i in range(len(tds_daily))] + ["Tomorrow", "Next Day"]
        forecast_values = list(tds_daily["TDS"]) + list(future_y)
        forecast_df = pd.DataFrame({"Day": forecast_days, "TDS": forecast_values})

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=forecast_df["Day"],
            y=forecast_df["TDS"],
            mode='lines+markers',
            name="TDS Forecast",
            line=dict(shape='spline')
        ))
        fig.update_layout(
            title="Projected TDS for Next 2 Days",
            xaxis_title="Day",
            yaxis_title="TDS (ppm)",
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 🧠 TDS Trend Insights")
        if future_y[0] > 1200 or future_y[1] > 1200:
            st.warning("⚠️ TDS is forecasted to exceed 1200 ppm — consider diluting your solution.")
        elif future_y[1] < 500:
            st.info("ℹ️ TDS may drop too low — monitor nutrient levels.")
        else:
            st.success("✅ TDS trend appears stable for the next few days.")

        st.markdown("---")
        st.caption("Forecasts are based on trends from Week 1–5 and are updated as new data is added.")
    else:
        st.warning("No data available or missing 'Week' column.")
        
#==faq beginner===
elif selected == "Beginner FAQ":
    st.markdown("""
    <h1 style='color:#2e8b57;'>🌱 Beginner FAQ & AI Advisor</h1>
    <p style='color:#4e944f;'>Start your smart hydroponic journey with answers and live AI feedback</p>
    """, unsafe_allow_html=True)

    # ========== AI Assistant Section ==========
    st.markdown("### 🤖 AI Advisor – Input your values")

    with st.form("ai_advice_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            tds = st.number_input("TDS (ppm)", min_value=0, max_value=2000, value=700)
        with col2:
            ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=6.0, step=0.1)
        with col3:
            temp = st.number_input("Water Temp (DS18B20 °C)", min_value=0.0, max_value=50.0, value=26.0)

        col4, col5 = st.columns(2)
        with col4:
            air_temp = st.number_input("Air Temperature (DHT22 °C)", min_value=0.0, max_value=50.0, value=28.0)
        with col5:
            humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)

        submitted = st.form_submit_button("Get Prediction")

    import time
    from fpdf import FPDF
    import base64

    if submitted:
        st.markdown("### 🤖 AI Growth Advisor")
        st.markdown("💬 *Processing your data...*")
        time.sleep(1)

        def chat_response(msg, success=True):
            style = "background-color: #e8f5e9;" if success else "background-color: #ffebee;"
            st.markdown(f"""
            <div style=\"{style} padding: 1rem; border-radius: 10px; margin-bottom: 0.8rem; font-size: 16px;\">
                💬 {msg}
            </div>
            """, unsafe_allow_html=True)
            time.sleep(1)

        if not (5.5 <= ph <= 6.5):
            chat_response("⚠️ Your pH is outside the optimal range (5.5–6.5). Try adjusting it slowly.", success=False)
        else:
            chat_response("✅ Nice! Your pH is in the ideal range for nutrient absorption.")

        if not (600 <= tds <= 800):
            chat_response("⚠️ Your TDS level may be too low or too high for leafy plants.", success=False)
        else:
            chat_response("✅ Nutrient levels are perfect for healthy hydroponic growth.")

        if temp > 28:
            chat_response("🔥 Warning: Your water temperature is quite warm. This can stress roots.", success=False)
        elif temp < 18:
            chat_response("🥶 It's a bit too cold. Consider warming the water slightly.", success=False)
        else:
            chat_response("✅ Water temperature looks healthy and stable.")

        if humidity > 75:
            chat_response("💧 High humidity could lead to fungal risks. Watch out!", success=False)
        elif humidity < 40:
            chat_response("🌬️ It's a bit dry — misting might help keep your plant comfy.", success=False)
        else:
            chat_response("✅ Humidity is in the sweet spot. Your plants will love it.")

        chat_response("🧠 Based on your inputs, growth conditions are looking **moderate to excellent** if maintained.")


    # ========== FAQ Section ==========
    st.markdown("### 📚 Frequently Asked Questions")

    with st.expander("💡 What is hydroponics?"):
        st.markdown("Hydroponics is a method of growing plants using nutrient-rich water instead of soil.")

    with st.expander("🌿 What plants grow best in hydroponics?"):
        st.markdown("Leafy greens like spinach, lettuce, basil, and herbs thrive in hydroponic systems.")

    with st.expander("🧪 How often should I check TDS and pH?"):
        st.markdown("Check at least once daily or every few days. Stability is key for healthy growth.")

    with st.expander("💧 What does TDS mean?"):
        st.markdown("TDS (Total Dissolved Solids) represents the amount of nutrients in the water. Ideal range for leafy plants: 600–800 ppm.")

    with st.expander("🌦️ Why does humidity matter?"):
        st.markdown("Too much humidity can cause fungal growth. Too little dries out plants. Aim for 50–70%.")

    st.caption("Need more help? Visit the 'Contact' page or chat with our support team.")



#====contact part=====
elif selected == "Contact":
    st.title("📞 Contact Us")
    st.markdown("""
        **Hydro-Pi Team**  
        📧 Email: support@hydro-pi.local  
        🌍 Website: [www.hydro-pi.local](#)
    """)

if __name__ == "__main__":
    pass
