import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
from keras.models import load_model
from keras.losses import MeanSquaredError

# Load saved model and scalers
model = load_model("lstm_model.h5", custom_objects={'mse': MeanSquaredError()})
scaler_x = joblib.load("scaler_x.pkl")
scaler_y = joblib.load("scaler_y.pkl")
last_30_days = pd.read_csv("last_30_days.csv", index_col=0)

st.set_page_config(page_title="Readmission Predictor", layout="centered")
st.markdown("<h2 style='text-align: center;'>🏥 Predict Patient Readmission Rate</h2>", unsafe_allow_html=True)

st.subheader("Enter Today's Hospital Metrics:")
# Input form
with st.form("new_day_form"):
    num_patients_admitted = st.number_input("Number of patients admitted", min_value=0, value=100)
    avg_length_of_stay = st.number_input("Average length of stay", min_value=0.0, value=5.0)
    avg_lab_result_score = st.number_input("Average lab result score", min_value=0.0, value=50.0)
    hospital_resource_utilization = st.number_input("Hospital resource utilization", min_value=0.0, max_value=1.0, value=0.5)
    submitted = st.form_submit_button("Predict Readmission")

# Run prediction
if submitted:
    new_day = {
        'num_patients_admitted': num_patients_admitted,
        'avg_length_of_stay': avg_length_of_stay,
        'avg_lab_result_score': avg_lab_result_score,
        'hospital_resource_utilization': hospital_resource_utilization
    }

    new_day_df = pd.DataFrame([new_day])

    # Combine with last 30 days
    combined = pd.concat([last_30_days, new_day_df], ignore_index=True)

    # Scale input
    X_scaled = scaler_x.transform(combined)
    X_input = X_scaled[-30:].reshape(1, 30, 4)

    # Predict and inverse scale
    y_pred_scaled = model.predict(X_input)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    st.success(f"📊 Predicted Readmission Rate: **{y_pred[0][0]:.4f}**")

    # Show input
    #st.markdown("#### Input Summary")
    #st.dataframe(new_day_df)



st.markdown("### 📈 30-Day Forecast")

# Forecast for next 30 days using current last_30_days
forecast_input = last_30_days.copy()

forecast_predictions = []

for i in range(30):
    # Scale input
    input_scaled = scaler_x.transform(forecast_input)
    X_input = input_scaled[-30:].reshape(1, 30, 4)

    # Predict and inverse scale
    y_pred_scaled = model.predict(X_input)
    y_pred = scaler_y.inverse_transform(y_pred_scaled)

    # Save prediction
    forecast_predictions.append(y_pred[0][0])

    # Create next day's features (dummy logic, can be improved)
    next_day = forecast_input.iloc[-1].copy()
    next_day['num_patients_admitted'] *= 1.01  # Slight increase
    next_day['avg_length_of_stay'] *= 0.99     # Slight decrease
    next_day['avg_lab_result_score'] *= 1.00
    next_day['hospital_resource_utilization'] *= 1.00

    forecast_input = pd.concat([forecast_input, pd.DataFrame([next_day])], ignore_index=True)

# Plot
fig, ax = plt.subplots()
ax.plot(range(1, 31), forecast_predictions, marker='o', linestyle='-')
ax.set_title("Predicted Readmission Rate for Next 30 Days")
ax.set_xlabel("Day")
ax.set_ylabel("Readmission Rate")
st.pyplot(fig)

# Show table
forecast_df = pd.DataFrame({
    "Day": range(1, 31),
    "Predicted Readmission Rate": np.round(forecast_predictions, 4)
})
st.dataframe(forecast_df)
