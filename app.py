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
        options=["Home", "About Us", "Historical Data", "Environment Monitor", "Growth Consistency", "Insights", "Contact"],
        icons=["house", "info-circle", "clock-history", "bar-chart", "activity", "lightbulb", "envelope"],
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
    st.success("🧬 Crop: *Spinach*  \n📅 Started: 3 weeks ago  \n⚙️ Method: Deep Water Culture")

    # 📸 GROWTH TIMELINE CAROUSEL (OPTION A)
    st.markdown("### 🌱 Growth Timeline Viewer")

    carousel_html = """
    <link rel="stylesheet" href="https://unpkg.com/swiper@10/swiper-bundle.min.css"/>
    <script src="https://unpkg.com/swiper@10/swiper-bundle.min.js"></script>

    <style>
    .swiper { width: 100%; height: 400px; border-radius: 15px; overflow: hidden; }
    .swiper-slide { display: flex; justify-content: center; align-items: center; background: #f0fff4; }
    .swiper-slide img {
        width: auto; max-width: 90%; max-height: 350px;
        border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    </style>

    <div class="swiper">
      <div class="swiper-wrapper">
        <div class="swiper-slide"><img src="growth/week1.png" alt="Week 1"></div>
        <div class="swiper-slide"><img src="growth/week2.png" alt="Week 2"></div>
        <div class="swiper-slide"><img src="growth/week3.png" alt="Week 3"></div>
      </div>
      <div class="swiper-pagination"></div>
      <div class="swiper-button-prev"></div>
      <div class="swiper-button-next"></div>
    </div>

    <script>
    const swiper = new Swiper('.swiper', {
      loop: true,
      pagination: { el: '.swiper-pagination' },
      navigation: {
        nextEl: '.swiper-button-next',
        prevEl: '.swiper-button-prev'
      }
    });
    </script>
    """

    components.html(carousel_html, height=450)

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

    # 🚀 CTA to Jump into Monitoring
    st.markdown("""
        <div style="margin-top: 2rem; background: #dfffe0; padding: 1.5rem; border-radius: 15px; text-align: center;">
            <h3 style="color: #1b5e20;">Ready to Grow Smarter?</h3>
            <p style="color: #2e7d32;">Check your plant environment now and get instant recommendations.</p>
            <a href="#Environment Monitor" style="background-color: #4CAF50; color: white; padding: 10px 20px;
                      text-decoration: none; border-radius: 10px; font-weight: bold;">🌿 Go to Live Monitor</a>
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
                df = pd.read_excel(uploaded_file, sheet_name="Sheet1")
                df['Week'] = df['Week'].ffill()
                st.success("File uploaded successfully!")

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
    st.subheader("Weekly Summary (Mean and Standard Deviation)")

    if 'Week' in df.columns:
        stat_table = df.groupby("Week")[['TDS', 'pH', 'DHT22 1', 'HUM 1', 'DHT 22 2', 'HUM 2', 'DS18B20']].agg(['mean', 'std'])
        st.dataframe(stat_table.style.format("{:.2f}"), height=350, use_container_width=True)
    else:
        st.warning("Week column not found — unable to compute weekly summary.")



    # ===== CORRELATION ANALYSIS =====
    st.subheader("🔗 Parameter Correlations")

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
            filtered_df['Growth_Score'] = ((filtered_df['Growth_Score'] - filtered_df['Growth_Score'].min()) / 
                                        (filtered_df['Growth_Score'].max() - filtered_df['Growth_Score'].min())) * 100
            
            st.line_chart(filtered_df.set_index('Time' if 'Time' in filtered_df.columns else filtered_df.index)['Growth_Score'])
            
            # Machine Learning Prediction
            if st.checkbox("Show Advanced Predictions", help="Show machine learning predictions vs actual growth"):
                X = filtered_df[['pH', 'TDS', 'DS18B20', 'HUM 1']]
                y = filtered_df['Growth_Score']
                
                model = RandomForestRegressor()
                model.fit(X, y)
                predictions = model.predict(X)
                
                pred_df = pd.DataFrame({
                    'Time': filtered_df['Time'] if 'Time' in filtered_df.columns else filtered_df.index,
                    'Actual': y,
                    'Predicted': predictions
                }).set_index('Time')
                
                st.line_chart(pred_df)
                st.metric("Prediction Accuracy", 
                         f"{100 - mean_squared_error(y, predictions, squared=False):.1f}%")
        else:
            st.warning("Missing required columns for growth score calculation")

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

        st.markdown("### 📈 Recent Sensor Trends (pH, TDS, Water Temp)")
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

        st.markdown("### 🔗 Correlation Heatmap")
        corr_matrix = df[['pH', 'TDS', 'DS18B20', 'DHT22 1', 'HUM 1', 'DHT 22 2', 'HUM 2']].corr().round(2)
        fig_corr = ff.create_annotated_heatmap(
            z=corr_matrix.values,
            x=list(corr_matrix.columns),
            y=list(corr_matrix.index),
            annotation_text=corr_matrix.values,
            colorscale='YlGnBu',
            showscale=True
        )
        fig_corr.update_layout(title_text="Sensor Correlation Heatmap", title_x=0.5)
        st.plotly_chart(fig_corr, use_container_width=True)

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
    st.title("Growth Consistency Analysis")
    daily_df = load_daily()

    # Debug: Show the loaded data
    st.write("Loaded Daily Data:", daily_df)

    if not daily_df.empty:
        st.subheader("Daily Variation Analysis")

        available_params = [col for col in ['Avg TDS', 'Avg pH', 'Avg DHT22 1', 'Avg HUM 1', 'Avg DS18B20'] if col in daily_df.columns]

        parameters = st.multiselect(
            "Select parameters to analyze",
            available_params,
            default=['Avg TDS', 'Avg pH'] if 'Avg TDS' in daily_df.columns and 'Avg pH' in daily_df.columns else available_params[:2]
        )

        if parameters:
            # Coefficient of variation
            cv_data = daily_df[parameters].std() / daily_df[parameters].mean()

            st.subheader("Consistency Metrics (Coefficient of Variation)")
            st.bar_chart(cv_data)

            st.info("""
            **Interpretation:**
            - Lower values = more consistent
            - Higher values = more variable
            - Ideal: Below 0.1 (10% variation)
            """)

            # Time series with rolling average
            if 'Day' in daily_df.columns:
                st.subheader("7-Day Moving Average")
                rolling_df = daily_df.set_index('Day')[parameters].rolling(7).mean()
                st.line_chart(rolling_df)
            else:
                st.warning("'Day' column not found for moving average.")

            # Inconsistency alerts
            st.subheader("Inconsistency Alerts")
            for param in parameters:
                std_dev = daily_df[param].std()
                mean_val = daily_df[param].mean()
                if mean_val != 0 and std_dev > (mean_val * 0.15):
                    st.warning(f"High variability in {param} (Std Dev: {std_dev:.2f})")
    else:
        st.warning("No daily data available. Please check your data source.")


# ============= OTHER PAGES =============

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

#==== INSIGHT===============
elif selected == "Insights":
    st.title("💡 Insights & Recommendations")
    st.info("Advanced insights coming in next update!")

#===== CONTACT=====
elif selected == "Contact":
    st.title("📞 Contact Us")
    st.markdown("""
        **Hydro-Pi Team**  
        📧 Email: support@hydro-pi.local  
        🌍 Website: [www.hydro-pi.local](#)
    """)

if __name__ == "__main__":
    pass
