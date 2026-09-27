import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="COVID-19 India Dashboard", page_icon="🦠", layout="wide")

# Title and Header
st.title("🦠 COVID-19 India Interactive Dashboard")
st.markdown("Analyzing trends, forecasting, and visualizing the impact of COVID-19 in India using **Prophet** & **Plotly**.")

# --- DATA LOADING & PROCESSING (Cached for speed) ---
@st.cache_data
def load_and_process_data():
    url_confirmed = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv'
    url_deaths = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_deaths_global.csv'
    url_recovered = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_recovered_global.csv'
    
    df_confirmed = pd.read_csv(url_confirmed)
    df_deaths = pd.read_csv(url_deaths)
    df_recovered = pd.read_csv(url_recovered)
    
    def process_data(df, metric_name):
        df_melted = pd.melt(df, id_vars=['Province/State', 'Country/Region', 'Lat', 'Long'], var_name='Date', value_name=metric_name)
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
    
    # Calculations
    df_india['Recovery_Rate'] = (df_india['Recovered'] / df_india['Confirmed']) * 100
    df_india['Mortality_Rate'] = (df_india['Deaths'] / df_india['Confirmed']) * 100
    df_india['Active'] = df_india['Confirmed'] - df_india['Deaths'] - df_india['Recovered']
    df_india['Daily_Confirmed'] = df_india['Confirmed'].diff()
    df_india['7-Day_Avg_Cases'] = df_india['Daily_Confirmed'].rolling(window=7).mean()
    
    # World data for comparison
    df_world = df_master.groupby('Date', as_index=False)[['Confirmed', 'Recovered']].sum()
    df_world['World_Recovery_Rate'] = (df_world['Recovered'] / df_world['Confirmed']) * 100
    
    return df_india, df_world

# --- PROPHET MODEL (Cached so it doesn't retrain on every click) ---
@st.cache_resource
def train_prophet_model(df):
    df_prophet = df[['Date', 'Confirmed']].rename(columns={'Date': 'ds', 'Confirmed': 'y'})
    model = Prophet(daily_seasonality=True)
    model.fit(df_prophet)
    future = model.make_future_dataframe(periods=7)
    forecast = model.predict(future)
    return model, forecast, df_prophet

with st.spinner('Fetching data from Johns Hopkins University...'):
    df_india, df_world = load_and_process_data()

with st.spinner('Training Prophet Forecasting Model...'):
    model, forecast, df_prophet = train_prophet_model(df_india)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📊 Navigation")
section = st.sidebar.radio("Select Dashboard Section", [
    "1. Overview & Trends", 
    "2. Rates & Global Comparison", 
    "3. Forecasting (Prophet)", 
    "4. Model Evaluation"
])

# --- DASHBOARD SECTIONS ---
if section == "1. Overview & Trends":
    st.header("📈 Historical Trends in India")
    
    col1, col2 = st.columns(2)
    with col1:
        fig_trends = px.line(df_india, x='Date', y=['Confirmed', 'Recovered', 'Deaths'], 
                             title='Cumulative Cases (Confirmed, Recovered, Deaths)')
        fig_trends.update_layout(hovermode="x unified")
        st.plotly_chart(fig_trends, use_container_width=True)
        
    with col2:
        fig_active = px.area(df_india.tail(100), x='Date', y=['Active', 'Recovered'], 
                             title='Active vs Recovered (Last 100 Days)')
        st.plotly_chart(fig_active, use_container_width=True)
        
    fig_daily = px.line(df_india.tail(100), x='Date', y=['Daily_Confirmed', '7-Day_Avg_Cases'], 
                        title='Daily New Cases vs 7-Day Rolling Average')
    st.plotly_chart(fig_daily, use_container_width=True)

elif section == "2. Rates & Global Comparison":
    st.header("📉 Recovery & Mortality Rates")
    
    fig_rates = px.line(df_india, x='Date', y=['Recovery_Rate', 'Mortality_Rate'], 
                        title='Recovery and Mortality Rates in India (%)')
    st.plotly_chart(fig_rates, use_container_width=True)
    
    st.header("🌍 India vs. World Recovery Rate")
    df_comparison = pd.merge(df_india[['Date', 'Recovery_Rate']], df_world[['Date', 'World_Recovery_Rate']], on='Date')
    df_plot_melted = df_comparison.melt(id_vars=['Date'], value_vars=['Recovery_Rate', 'World_Recovery_Rate'], 
                                        var_name='Region', value_name='Recovery Rate (%)')
    fig_comp = px.line(df_plot_melted.tail(100), x='Date', y='Recovery Rate (%)', color='Region', 
                       title='Recovery Rate: India vs. World Average (%)')
    st.plotly_chart(fig_comp, use_container_width=True)

elif section == "3. Forecasting (Prophet)":
    st.header("🔮 7-Day Future Prediction (Prophet)")
    
    historical_data = df_prophet.tail(30)
    forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(7)

    fig_forecast = go.Figure()
    fig_forecast.add_trace(go.Scatter(x=historical_data['ds'], y=historical_data['y'], mode='lines+markers', name='Historical', line=dict(color='blue')))
    fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'], y=forecast_data['yhat'], mode='lines+markers', name='7-Day Forecast', line=dict(color='red', dash='dash')))
    fig_forecast.add_trace(go.Scatter(x=forecast_data['ds'].tolist() + forecast_data['ds'][::-1].tolist(),
                                      y=forecast_data['yhat_upper'].tolist() + forecast_data['yhat_lower'][::-1].tolist(),
                                      fill='toself', fillcolor='rgba(255,0,0,0.1)', line=dict(color='rgba(255,255,255,0)'), name='Confidence Interval'))
    fig_forecast.update_layout(title='Historical Data + 7-Day Future Prediction', hovermode="x unified", template="plotly_white")
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.subheader("Prophet Decomposition")
    fig_components = model.plot_components(forecast)
    st.pyplot(fig_components)

elif section == "4. Model Evaluation":
    st.header("🎯 Model Evaluation Metrics")
    st.markdown("Evaluating how well the Prophet model fits the historical data.")
    
    df_eval = pd.merge(df_prophet, forecast[['ds', 'yhat']], on='ds')
    r2 = r2_score(df_eval['y'], df_eval['yhat'])
    mae = mean_absolute_error(df_eval['y'], df_eval['yhat'])
    rmse = np.sqrt(mean_squared_error(df_eval['y'], df_eval['yhat']))
    df_eval_nz = df_eval[df_eval['y'] > 0]
    mape = np.mean(np.abs((df_eval_nz['y'] - df_eval_nz['yhat']) / df_eval_nz['y'])) * 100
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{r2:.4f}")
    col2.metric("MAE", f"{mae:,.0f}")
    col3.metric("RMSE", f"{rmse:,.0f}")
    col4.metric("MAPE", f"{mape:.2f}%")
    
    st.markdown("""
    **Metric Definitions:**
    - **R² Score**: Represents the proportion of variance for the dependent variable that's explained by the model (Close to 1.0 is excellent).
    - **MAE (Mean Absolute Error)**: The average absolute difference between predicted and actual values.
    - **RMSE (Root Mean Squared Error)**: Standard deviation of the prediction errors.
    - **MAPE (Mean Absolute Percentage Error)**: Average percentage error of the predictions.
    """)