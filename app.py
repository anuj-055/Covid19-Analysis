import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

st.set_page_config(page_title="COVID-19 India Retrospective Analysis", page_icon="🦠", layout="wide")

st.title("🦠 COVID-19 India — Retrospective Analysis Dashboard")
st.markdown("""
A **post-pandemic retrospective** analysis of COVID-19 trends in India.
This dashboard analyzes historical data from the Johns Hopkins University repository 
to understand how the pandemic evolved, evaluate forecasting accuracy through **backtesting**, 
and draw insights from the crisis that shaped the nation.
""")

# --- DATA LOADING ---
@st.cache_data
def load_and_process_data():
    url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'

    df_confirmed = pd.read_csv(url_confirmed)
    df_deaths = pd.read_csv(url_deaths)
    df_recovered = pd.read_csv(url_recovered)

    def process_data(df, metric_name):
        df_melted = pd.melt(df, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'],
                            var_name='Date', value_name=metric_name)
        df_melted['Date'] = pd.to_datetime(df_melted['Date'])
        df_grouped = df_melted.groupby(['Country/Region', 'Date'], as_index=False)[metric_name].sum()
        return df_grouped

    df_conf_long = process_data(df_confirmed, 'Confirmed')
    df_death_long = process_data(df_deaths, 'Deaths')
    df_recov_long = process_data(df_recovered, 'Recovered')

    df_master = pd.merge(df_conf_long, df_death_long, on=['Country/Region', 'Date'])
    df_master = pd.merge(df_master, df_recov_long, on=['Country/Region', 'Date'])

    df_india = df_master[df_master['Country/Region'] == 'India'].copy()
    df_india = df_india.sort_values('Date').reset_index(drop=True)

    # --- KEY FIX: JHU stopped tracking Recovered after ~March 2022 ---
    # We find the last date where Recovered was actually being updated (non-zero change)
    df_india['Recovered_Diff'] = df_india['Recovered'].diff()
    last_valid_recovered_date = df_india[df_india['Recovered_Diff'] > 0]['Date'].max()

    # Create a version with valid recovered data only
    df_india_valid_recovered = df_india[df_india['Date'] <= last_valid_recovered_date].copy()
    df_india_valid_recovered['Recovery_Rate'] = (df_india_valid_recovered['Recovered'] / df_india_valid_recovered['Confirmed']) * 100
    df_india_valid_recovered['Mortality_Rate'] = (df_india_valid_recovered['Deaths'] / df_india_valid_recovered['Confirmed']) * 100

    # For full timeline, calculate Mortality Rate only (Deaths data is reliable throughout)
    df_india['Mortality_Rate'] = (df_india['Deaths'] / df_india['Confirmed']) * 100
    df_india['Daily_Confirmed'] = df_india['Confirmed'].diff()
    df_india['7-Day_Avg_Cases'] = df_india['Daily_Confirmed'].rolling(window=7).mean()

    # Last known recovered count
    last_recovered_count = int(df_india_valid_recovered['Recovered'].iloc[-1])

    return df_india, df_india_valid_recovered, last_recovered_count, last_valid_recovered_date

with st.spinner('Loading data from Johns Hopkins University...'):
    df_india, df_valid, last_recovered, last_valid_date = load_and_process_data()

# --- SIDEBAR ---
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Select Section", [
    "1. Pandemic Overview",
    "2. Wave Analysis",
    "3. Forecast Backtesting",
    "4. Model Evaluation"
])

# ===================== SECTION 1 =====================
if section == "1. Pandemic Overview":
    st.header("📈 Pandemic Overview (Jan 2020 – Mar 2023)")

    # Final cumulative numbers
    final = df_india.iloc[-1]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Confirmed", f"{final['Confirmed']:,.0f}")
    col2.metric("Total Deaths", f"{final['Deaths']:,.0f}")
    col3.metric("Last Known Recovered", f"{last_recovered:,.0f}",
                help=f"JHU stopped tracking recovered cases after {last_valid_date.strftime('%b %Y')}")
    col4.metric("Final Mortality Rate", f"{final['Mortality_Rate']:.2f}%")

    st.info(f"⚠️ **Data Note:** Johns Hopkins University stopped tracking 'Recovered' cases globally after **{last_valid_date.strftime('%B %Y')}**. "
            f"The recovered count shown is the last reliably recorded figure. Recovery and mortality rate charts below only display data up to that date.")

    st.markdown("---")

    # Cumulative Confirmed & Deaths (full timeline — these are reliable)
    fig_cumulative = px.line(df_india, x='Date', y=['Confirmed', 'Deaths'],
                             title='Cumulative Confirmed Cases & Deaths (Full Timeline)',
                             labels={'value': 'Count', 'variable': 'Metric'})
    fig_cumulative.update_layout(hovermode="x unified")
    st.plotly_chart(fig_cumulative, use_container_width=True)

    # Recovery & Mortality Rate (ONLY valid period)
    st.subheader("Recovery & Mortality Rate (Valid Data Period)")
    fig_rates = px.line(df_valid, x='Date', y=['Recovery_Rate', 'Mortality_Rate'],
                        title=f'Recovery & Mortality Rates (up to {last_valid_date.strftime("%b %Y")})',
                        labels={'value': 'Percentage (%)', 'variable': 'Rate'})
    fig_rates.update_layout(hovermode="x unified")
    st.plotly_chart(fig_rates, use_container_width=True)

# ===================== SECTION 2 =====================
elif section == "2. Wave Analysis":
    st.header("🌊 Wave Analysis — Daily Cases & 7-Day Average")
    st.markdown("This chart shows the **daily reported cases** (bars) against the **7-day rolling average** (line) "
                "to clearly identify the major waves of the pandemic in India.")

    # Filter out negative daily values (data corrections)
    df_daily = df_india[df_india['Date'] >= '2020-03-01'].copy()
    df_daily['Daily_Confirmed'] = df_daily['Daily_Confirmed'].clip(lower=0)
    df_daily['7-Day_Avg_Cases'] = df_daily['Daily_Confirmed'].rolling(window=7).mean()

    fig_daily = go.Figure()
    fig_daily.add_trace(go.Bar(
        x=df_daily['Date'], y=df_daily['Daily_Confirmed'],
        name='Daily New Cases',
        marker_color='rgba(99, 110, 250, 0.4)'
    ))
    fig_daily.add_trace(go.Scatter(
        x=df_daily['Date'], y=df_daily['7-Day_Avg_Cases'],
        name='7-Day Rolling Average',
        line=dict(color='red', width=2.5)
    ))
    fig_daily.update_layout(
        title='Daily New Confirmed Cases with 7-Day Moving Average',
        xaxis_title='Date', yaxis_title='Cases',
        hovermode="x unified", template="plotly_white", barmode='overlay'
    )
    st.plotly_chart(fig_daily, use_container_width=True)

    # Wave annotations
    st.markdown("""
    **Major Waves Identified:**
    - 🟡 **Wave 1 (Jul–Nov 2020):** Peak ~97K daily cases in September 2020
    - 🔴 **Wave 2 (Mar–Jun 2021):** Devastating Delta wave, peak ~414K daily cases in May 2021
    - 🟠 **Wave 3 (Jan–Feb 2022):** Omicron variant, peak ~347K daily cases in January 2022
    """)

# ===================== SECTION 3 =====================
elif section == "3. Forecast Backtesting":
    st.header("🔮 Forecast Backtesting (Not Future Prediction)")
    st.markdown("""
    Since the pandemic is over, predicting the future has no real-world value. 
    Instead, we perform **backtesting**: we train the Prophet model on data **up to September 2021** 
    (before the devastating Wave 2 peak ended) and ask it to **predict the next 6 months** 
    (Oct 2021 – Mar 2022). We then compare its predictions against what **actually happened**.
    
    This tells us: *Could the model have foreseen the Omicron wave?*
    """)

    # Prepare data
    df_prophet_full = df_india[['Date', 'Confirmed']].rename(columns={'Date': 'ds', 'Confirmed': 'y'})

    # Split: Train up to 2021-09-01, Test = next 6 months
    train_end = '2021-09-01'
    test_end = '2022-03-01'

    df_train = df_prophet_full[df_prophet_full['ds'] <= train_end].copy()
    df_actual_test = df_prophet_full[(df_prophet_full['ds'] > train_end) & (df_prophet_full['ds'] <= test_end)].copy()

    with st.spinner('Training Prophet on data up to Sep 2021...'):
        model = Prophet(daily_seasonality=True, yearly_seasonality=True)
        model.fit(df_train)

        # Predict 180 days into the future from train end
        future = model.make_future_dataframe(periods=180)
        forecast = model.predict(future)

    # Filter forecast to test period
    df_forecast_test = forecast[(forecast['ds'] > train_end) & (forecast['ds'] <= test_end)].copy()

    # Plot
    fig_bt = go.Figure()

    # Training data
    fig_bt.add_trace(go.Scatter(
        x=df_train['ds'], y=df_train['y'],
        mode='lines', name='Training Data (up to Sep 2021)',
        line=dict(color='blue', width=2)
    ))

    # Actual test data
    fig_bt.add_trace(go.Scatter(
        x=df_actual_test['ds'], y=df_actual_test['y'],
        mode='lines', name='Actual (Oct 2021 – Mar 2022)',
        line=dict(color='green', width=2.5)
    ))

    # Predicted test data
    fig_bt.add_trace(go.Scatter(
        x=df_forecast_test['ds'], y=df_forecast_test['yhat'],
        mode='lines', name='Prophet Prediction',
        line=dict(color='red', dash='dash', width=2)
    ))

    # Confidence interval
    fig_bt.add_trace(go.Scatter(
        x=df_forecast_test['ds'].tolist() + df_forecast_test['ds'][::-1].tolist(),
        y=df_forecast_test['yhat_upper'].tolist() + df_forecast_test['yhat_lower'][::-1].tolist(),
        fill='toself', fillcolor='rgba(255,0,0,0.1)',
        line=dict(color='rgba(255,255,255,0)'),
        name='95% Confidence Interval'
    ))

    # Vertical line at train/test split
    fig_bt.add_vline(x=train_end, line_dash="dot", line_color="gray",
                     annotation_text="Train/Test Split")

    fig_bt.update_layout(
        title='Backtesting: Can Prophet Predict the Post-Wave-2 Trajectory?',
        xaxis_title='Date', yaxis_title='Cumulative Confirmed Cases',
        hovermode="x unified", template="plotly_white"
    )
    st.plotly_chart(fig_bt, use_container_width=True)

    st.warning("💡 **Key Insight:** Prophet predicted a continued gradual rise based on Wave 1 patterns, "
               "but it **could not foresee** the massive Omicron-driven Wave 3 spike in Jan 2022. "
               "This is a fundamental limitation of time-series models — they cannot predict **unprecedented events** "
               "(new variants, policy changes) that have no precedent in the training data.")

    # Store for evaluation section
    st.session_state['df_train'] = df_train
    st.session_state['df_actual_test'] = df_actual_test
    st.session_state['df_forecast_test'] = df_forecast_test

# ===================== SECTION 4 =====================
elif section == "4. Model Evaluation":
    st.header("🎯 Model Evaluation — Backtesting Metrics")
    st.markdown("Evaluating how well Prophet's predictions matched reality during the **Oct 2021 – Mar 2022** test period.")

    if 'df_actual_test' not in st.session_state:
        st.warning("⚠️ Please visit **Section 3 (Forecast Backtesting)** first to generate the predictions.")
    else:
        df_actual = st.session_state['df_actual_test']
        df_pred = st.session_state['df_forecast_test']

        # Merge on date
        df_eval = pd.merge(df_actual, df_pred[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

        r2 = r2_score(df_eval['y'], df_eval['yhat'])
        mae = mean_absolute_error(df_eval['y'], df_eval['yhat'])
        rmse = np.sqrt(mean_squared_error(df_eval['y'], df_eval['yhat']))

        # MAPE — only on meaningful values (y > 10000 to avoid early-data distortion)
        df_eval_filtered = df_eval[df_eval['y'] > 10000]
        if len(df_eval_filtered) > 0:
            mape = np.mean(np.abs((df_eval_filtered['y'] - df_eval_filtered['yhat']) / df_eval_filtered['y'])) * 100
        else:
            mape = float('nan')

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{r2:.4f}", help="How much variance the model explains (1.0 = perfect)")
        col2.metric("MAE", f"{mae/1e6:.2f}M", help="Average absolute error in millions of cases")
        col3.metric("RMSE", f"{rmse/1e6:.2f}M", help="Root mean squared error in millions")
        col4.metric("MAPE", f"{mape:.2f}%", help="Mean absolute percentage error (filtered for y > 10K)")

        st.markdown("---")

        st.markdown("""
        ### How to Read These Metrics:
        | Metric | What It Means | Good Value |
        |--------|--------------|------------|
        | **R²** | % of variance explained by model | Close to 1.0 |
        | **MAE** | Average error per prediction | Lower is better |
        | **RMSE** | Penalizes large errors more than MAE | Lower is better |
        | **MAPE** | Average % error | < 10% is excellent |
        
        > **Note:** MAPE is calculated only on dates where confirmed cases exceeded 10,000. 
        > Including early-pandemic days (1–10 cases) would inflate MAPE to absurd values 
        > (e.g., predicting 5 instead of 2 = 150% error on a single day).
        """)

        # Residual plot
        df_eval['Residual'] = df_eval['y'] - df_eval['yhat']
        fig_residual = px.scatter(df_eval, x='ds', y='Residual',
                                  title='Prediction Residuals Over Time (Actual – Predicted)',
                                  labels={'ds': 'Date', 'Residual': 'Residual (Actual - Predicted)'})
        fig_residual.add_hline(y=0, line_dash="dash", line_color="red")
        fig_residual.update_layout(template="plotly_white")
        st.plotly_chart(fig_residual, use_container_width=True)
