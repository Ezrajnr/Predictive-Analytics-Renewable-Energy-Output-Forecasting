import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Page Configuration
st.set_page_config(page_title="Nigerian Renewable Energy Forecasting", layout="wide")
st.title("⚡ Predictive Analytics: Renewable Energy Output Forecasting Dashboard")
st.markdown("Interactive analysis of meteorological variables and machine learning forecasting performance.")

# Sidebar Controls
st.sidebar.header("Data Parameters")
sample_hours = st.sidebar.slider("Number of Sample Hours to Simulate", min_value=1000, max_value=8760, value=2000, step=500)
rf_trees = st.sidebar.slider("Random Forest Estimators", min_value=10, max_value=200, value=50, step=10)

# 1. Dataset Generation Function
@st.cache_data
def load_data(hours):
    np.random.seed(42)
    date_range = pd.date_range(start="2025-01-01", periods=hours, freq="h")
    hours_arr = date_range.hour.to_numpy()

    solar_irradiance = np.clip(np.sin(np.pi * (hours_arr - 6) / 12) * 900 + np.random.normal(0, 50, hours), 0, None)
    solar_irradiance[hours_arr < 6] = 0
    solar_irradiance[hours_arr > 18] = 0

    wind_speed = np.clip(np.random.normal(6.5, 2.5, hours), 0.5, 18.0)
    temperature = 22 + 12 * np.sin(np.pi * (hours_arr - 8) / 12) + np.random.normal(0, 1.5, hours)
    humidity = np.clip(80 - (temperature - 20) * 2 + np.random.normal(0, 5, hours), 20, 95)

    solar_output = (solar_irradiance * 0.18 * 1.5) - (0.004 * (temperature - 25) * solar_irradiance)
    wind_output = 0.5 * 1.225 * (wind_speed ** 3) * 0.35 * 0.1
    energy_output = np.clip(solar_output + wind_output + np.random.normal(0, 10, hours), 0, None)

    return pd.DataFrame({
        'Timestamp': date_range,
        'Solar_Irradiance': solar_irradiance,
        'Wind_Speed': wind_speed,
        'Temperature': temperature,
        'Humidity': humidity,
        'Energy_Output': energy_output
    }).set_index('Timestamp')

df = load_data(sample_hours)

# Data Preview & Summary Stats
st.subheader("📊 Dataset Overview")
col1, col2 = st.columns([2, 1])

with col1:
    st.dataframe(df.head(10), use_container_width=True)

with col2:
    st.metric(label="Total Hours Recorded", value=len(df))
    st.metric(label="Peak Energy Output (kW)", value=f"{df['Energy_Output'].max():.2f}")
    st.metric(label="Average Solar Irradiance (W/m²)", value=f"{df['Solar_Irradiance'].mean():.2f}")

# Objective I: Exploratory Analysis
st.divider()
st.subheader("Objective I: Correlation Heatmap & Exploratory Analysis")

col_left, col_right = st.columns(2)

with col_left:
    st.write("### Correlation Matrix")
    fig_corr, ax_corr = plt.subplots(figsize=(6, 4))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax_corr)
    st.pyplot(fig_corr)

with col_right:
    st.write("### Energy vs. Meteorological Variables")
    selected_var = st.selectbox("Select Variable to Plot against Energy Output:", ['Solar_Irradiance', 'Wind_Speed', 'Temperature', 'Humidity'])
    fig_scatter, ax_scatter = plt.subplots(figsize=(6, 4))
    sns.scatterplot(data=df, x=selected_var, y='Energy_Output', alpha=0.4, color='teal', ax=ax_scatter)
    st.pyplot(fig_scatter)

# Data Preprocessing & Training
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
train_size = int(len(scaled_data) * 0.8)

train_data, test_data = scaled_data[:train_size], scaled_data[train_size:]
X_train, y_train = train_data[:, :-1], train_data[:, -1]
X_test, y_test = test_data[:, :-1], test_data[:, -1]

# Model Training
rf_model = RandomForestRegressor(n_estimators=rf_trees, random_state=42)
rf_model.fit(X_train, y_train)
pred_rf = rf_model.predict(X_test)

svr_model = SVR(kernel='rbf', C=1.0)
svr_model.fit(X_train, y_train)
pred_svr = svr_model.predict(X_test)

# Metrics Calculation
def get_metrics(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R² Score": r2_score(y_true, y_pred)
    }

metrics_df = pd.DataFrame([
    {"Model": "Random Forest", **get_metrics(y_test, pred_rf)},
    {"Model": "Support Vector Regression (SVR)", **get_metrics(y_test, pred_svr)}
])

# Objectives II & III: Modeling & Performance Evaluation
st.divider()
st.subheader("Objectives II & III: Comparative Model Evaluation")

st.write("### Performance Metrics Summary")
st.dataframe(metrics_df, use_container_width=True)

st.write("### Comparative Forecast Window")
window = st.slider("Select Forecast Display Window (Hours)", min_value=24, max_value=240, value=96)

fig_eval, ax_eval = plt.subplots(figsize=(10, 4))
ax_eval.plot(y_test[:window], label='Actual Output', color='black', linewidth=2)
ax_eval.plot(pred_rf[:window], label='Random Forest', linestyle='--')
ax_eval.plot(pred_svr[:window], label='SVR', linestyle=':')
ax_eval.set_ylabel("Normalized Energy Output")
ax_eval.set_xlabel("Hours")
ax_eval.legend()
st.pyplot(fig_eval)
