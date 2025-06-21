# ============= ENVIRONMENT MONITOR PAGE =============
elif selected == "Environment Monitor":
    st.title("Environmental Monitoring Dashboard")
    weekly_df = load_weekly()

    # Debug: Show the loaded data
    st.write("Loaded Weekly Data:", weekly_df)

    if not weekly_df.empty:
        if 'Week' in weekly_df.columns:
            selected_week = st.selectbox("Select Week", weekly_df['Week'].unique())
            week_data = weekly_df[weekly_df['Week'] == selected_week].iloc[0]

            # Metrics
            col1, col2, col3 = st.columns(3)
            if 'Avg TDS' in week_data:
                col1.metric("Weekly Avg TDS", f"{week_data['Avg TDS']:.1f} ppm")
            if 'Avg pH' in week_data:
                col2.metric("Weekly Avg pH", f"{week_data['Avg pH']:.2f}")
            if 'Avg DS18B20' in week_data:
                col3.metric("Avg Water Temp", f"{week_data['Avg DS18B20']:.1f}°C")

            # Weekly trends
            st.subheader("Weekly Trends Over Time")
            plot_cols = [col for col in ['Avg TDS', 'Avg pH', 'Avg DS18B20'] if col in weekly_df.columns]
            if plot_cols:
                fig, ax = plt.subplots(figsize=(10, 6))
                weekly_df.set_index('Week')[plot_cols].plot(ax=ax)
                st.pyplot(fig)
            else:
                st.warning("No columns available for trend plotting.")

            # Environmental stability
            st.subheader("Environmental Stability")
            stability_data = weekly_df[plot_cols].std().reset_index()
            stability_data.columns = ['Parameter', 'Std Dev']
            st.bar_chart(stability_data.set_index('Parameter'))
        else:
            st.error("'Week' column not found in weekly data.")
    else:
        st.warning("No weekly data available. Please check your data source.")

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
