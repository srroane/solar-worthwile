from pathlib import Path

import pandas as pd
import streamlit as st

from api_integration import (
    geocode_city,
    get_weather_forecast,
    get_solar_production,
)
from solar_analytics import (
    build_demo_model,
    calculate_economics,
    predict_solar_output,
)


st.set_page_config(
    page_title="Solar Decision Engine",
    page_icon="☀️",
    layout="wide"
)


# --------------------------------------------------
# CONFIG
# --------------------------------------------------
FORECAST_DAYS = 7
BASE_KWP = 8.0
MODEL_PATH = Path("solar_rf_model.joblib")


# --------------------------------------------------
# SIDEBAR INPUTS
# --------------------------------------------------
with st.sidebar:
    st.header("Your inputs")

    city = st.text_input("City", value="Lausanne")

    st.subheader("Solar panel system")
    kwp = st.number_input("Installed capacity (kWp)", min_value=0.5, value=8.0, step=0.5)
    tilt = st.slider("Tilt / declination (°)", 0, 90, 30)
    azimuth = st.slider("Azimuth (°): -90 = East, 0 = South, 90 = West", -180, 180, 0)

    st.subheader("Financial inputs")
    electricity_price = st.number_input("Electricity price (CHF / kWh)", value=0.30, step=0.01, format="%.2f")
    feed_in_tariff = st.number_input("Feed-in tariff (CHF / kWh)", value=0.08, step=0.01, format="%.2f")
    install_cost = st.number_input("Installation cost (CHF)", value=20000, step=500)
    maintenance_cost = st.number_input("Yearly maintenance cost (CHF)", value=200, step=50)
    annual_consumption = st.number_input("Annual electricity consumption (kWh)", value=4500, step=100)

    run = st.button("Run analysis", type="primary", use_container_width=True)


# --------------------------------------------------
# ANALYSIS
# --------------------------------------------------
if run:
    try:
        with st.spinner("Fetching weather, solar and running ML..."):
            location = geocode_city(city)
            weather = get_weather_forecast(location["latitude"], location["longitude"], days=FORECAST_DAYS)

            try:
                solar_forecast = get_solar_production(
                    location["latitude"], location["longitude"], tilt, azimuth, kwp
                )
            except Exception as exc:
                solar_forecast = None
                st.warning(f"Forecast.Solar unavailable ({exc}). Using ML only.")

            if not MODEL_PATH.exists():
                build_demo_model(model_path=MODEL_PATH)

            production = predict_solar_output(weather, model_path=MODEL_PATH)
            production["solar_output_kwh"] *= kwp / BASE_KWP

            consumption = pd.Series(annual_consumption / 8760.0, index=production.index)

            econ = calculate_economics(
                production["solar_output_kwh"],
                {"retail_rate": electricity_price, "feed_in_tariff": feed_in_tariff},
                consumption,
            )

            scale = 365.0 / FORECAST_DAYS
            annual_revenue = econ["total_revenue"] * scale
            annual_net = annual_revenue - maintenance_cost
            payback_years = (install_cost / annual_net) if annual_net > 0 else float("inf")

            st.session_state["results"] = {
                "location": location,
                "production": production,
                "solar_forecast": solar_forecast,
                "econ": econ,
                "annual_production": econ["total_production_kwh"] * scale,
                "annual_savings": econ["bill_savings"] * scale,
                "annual_export": econ["feed_in_revenue"] * scale,
                "annual_revenue": annual_revenue,
                "annual_net": annual_net,
                "payback_years": payback_years,
                "install_cost": install_cost,
                "maintenance_cost": maintenance_cost,
                "annual_consumption": annual_consumption,
                "kwp": kwp,
            }

        st.success(f"Analysis complete for {location['name']}, {location['country']}.")
    except Exception as exc:
        st.error(f"Could not complete analysis: {exc}")

results = st.session_state.get("results")


# --------------------------------------------------
# NAVIGATION
# --------------------------------------------------
page = st.radio(
    "",
    ["Home", "Financial Results", "Graphs & Recommendation"],
    horizontal=True
)


# --------------------------------------------------
# PAGE 1: HOME
# --------------------------------------------------
if page == "Home":

    st.markdown("""
    <style>
    .hero {
        padding: 60px;
        border-radius: 28px;
        background: linear-gradient(135deg, #fff7d6, #e6f4ff);
        text-align: center;
        margin-bottom: 40px;
    }

    .title {
        font-size: 50px;
        font-weight: 800;
        color: #111827;
    }

    .subtitle {
        font-size: 20px;
        color: #4b5563;
        margin-top: 18px;
    }

    .card {
        padding: 28px;
        border-radius: 22px;
        background-color: white;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.07);
        margin-bottom: 25px;
    }

    .card h3 {
        color: #111827;
    }

    .card p {
        color: #374151;
        font-size: 16px;
    }

    .section-box {
        padding: 28px;
        border-radius: 22px;
        background-color: white;
        box-shadow: 0px 4px 18px rgba(0,0,0,0.07);
        margin-top: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="title">Should you invest in solar panels?</div>
        <div class="subtitle">
            Solar Decision Engine helps users evaluate if installing solar panels
            is financially and environmentally worth it.
        </div>
    </div>
    """, unsafe_allow_html=True)

    if results:
        prod_text = f"~ {results['annual_production']:,.0f} kWh / year estimated for your system."
        if results["payback_years"] != float("inf"):
            profit_text = (
                f"~ {results['annual_net']:,.0f} CHF / year net "
                f"(payback ≈ {results['payback_years']:.1f} years)."
            )
        else:
            profit_text = "Net yearly value is not positive, payback not reached."
        battery_text = (
            f"Self-consumption rate: {results['econ']['self_consumption_rate']*100:.0f} %. "
            "A battery would lift this further."
        )
    else:
        prod_text = "Enter your inputs in the sidebar and click Run analysis."
        profit_text = "Costs, savings and ROI will appear here."
        battery_text = "Self-consumption results will appear here."

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(f"""
        <div class="card">
            <h3>Solar Production</h3>
            <p>{prod_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="card">
            <h3>Profitability</h3>
            <p>{profit_text}</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="card">
            <h3>Battery Analysis</h3>
            <p>{battery_text}</p>
        </div>
        """, unsafe_allow_html=True)


# --------------------------------------------------
# PAGE 2: FINANCIAL RESULTS
# --------------------------------------------------
elif page == "Financial Results":

    st.title("Financial Results")

    if not results:
        st.info("Run an analysis from the sidebar to see your financial results.")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total installation cost", "—")
        with col2:
            st.metric("Yearly savings", "—")
        with col3:
            st.metric("Payback period", "—")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total installation cost", f"{results['install_cost']:,.0f} CHF")
        with col2:
            st.metric("Yearly savings (net)", f"{results['annual_net']:,.0f} CHF")
        with col3:
            payback = results["payback_years"]
            payback_label = f"{payback:.1f} years" if payback != float("inf") else "Not reached"
            st.metric("Payback period", payback_label)

        st.subheader("Breakdown")
        st.write(f"- Bill savings (self-consumed energy): **{results['annual_savings']:,.0f} CHF / year**")
        st.write(f"- Feed-in revenue (exported energy): **{results['annual_export']:,.0f} CHF / year**")
        st.write(f"- Maintenance cost: **-{results['maintenance_cost']:,.0f} CHF / year**")
        st.caption(
            "Annual figures are extrapolated from a 7-day weather forecast "
            "(scaled by 365 / 7). Real-world results vary by season."
        )


# --------------------------------------------------
# PAGE 3: GRAPHS & RECOMMENDATION
# --------------------------------------------------
elif page == "Graphs & Recommendation":

    st.title("Graphs and Final Recommendation")

    if not results:
        st.info("Run an analysis from the sidebar to see your graphs.")
        st.metric("Annual production", "—")
        st.metric("Energy coverage", "—")
        st.metric("Self-consumption rate", "—")
    else:
        econ = results["econ"]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Annual production", f"{results['annual_production']:,.0f} kWh")
        with col2:
            st.metric("Energy coverage (self-sufficiency)", f"{econ['self_sufficiency_rate']*100:.0f} %")
        with col3:
            st.metric("Self-consumption rate", f"{econ['self_consumption_rate']*100:.0f} %")

        st.subheader("ML-predicted hourly solar output (next 7 days)")
        st.line_chart(results["production"]["solar_output_kwh"])

        if results["solar_forecast"] is not None:
            st.subheader("Forecast.Solar daily production estimate")
            st.bar_chart(results["solar_forecast"]["production_kwh"])

        st.subheader("Production vs. consumption (hourly, 7 days)")
        comparison = pd.DataFrame({
            "production_kwh": econ["hourly"]["production_kwh"],
            "consumption_kwh": econ["hourly"]["consumption_kwh"],
        })
        st.area_chart(comparison)

        st.subheader("Recommendation")
        payback = results["payback_years"]
        if payback == float("inf"):
            st.error(
                "Under the current inputs, the system does not generate "
                "positive net yearly value. Investing is not recommended without "
                "changing assumptions (lower install cost, larger system, "
                "or better orientation)."
            )
        elif payback <= 10:
            st.success(f"Strong case: payback in about {payback:.1f} years. Investing is recommended.")
        elif payback <= 20:
            st.warning(f"Borderline case: payback in about {payback:.1f} years. Worth considering for long-term owners.")
        else:
            st.error(f"Long payback (~{payback:.1f} years). Investing is not recommended under the current inputs.")
