
# 🏥 Predict Patient Readmission Rate - Streamlit App

This Streamlit app allows you to predict the patient **readmission rate** based on recent hospital metrics using a trained **LSTM model**. It also provides a 30-day forecast and visual insights.

---

## 📌 Features

- Upload or use default hospital metrics
- Input new day's values: 
  - Number of patients admitted
  - Average length of stay
  - Average lab result score
  - Hospital resource utilization
- Predict today's readmission rate using a trained LSTM model
- Generate and visualize 30-day readmission rate forecast
- Downloadable prediction table

---

## 📷 App Screenshots

### 🎯 Single-Day Prediction
![Readmission Prediction UI](./Screenshot%202025-06-27%20105111.png)

### 📊 30-Day Forecast Plot
![30 Day Forecast](./Screenshot%202025-06-27%20105021.png)

---

## 🛠️ Requirements
```bash
pip install streamlit pandas numpy matplotlib scikit-learn tensorflow joblib
```

---

## 🚀 How to Run the App
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/readmission-forecast-app.git
   cd readmission-forecast-app
   ```

2. Ensure the following files are in the repo root:
   - `app.py`
   - `lstm_model.h5`
   - `scaler_x.pkl`
   - `scaler_y.pkl`
   - `last_30_days.csv`

3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

---

## 📁 Project Structure
```
readmission-forecast-app/
├── app.py
├── lstm_model.h5
├── scaler_x.pkl
├── scaler_y.pkl
├── last_30_days.csv
```

---

## 🎯 Project Aim & Real-World Impact

The aim of this project is to proactively identify potential hospital readmissions using predictive analytics. By leveraging historical patient data and a deep learning model (LSTM), this solution can assist healthcare providers in making data-driven decisions.

### 🧠 Theoretical Background
Patient readmission prediction is a crucial task in healthcare informatics. Hospital readmissions are not only costly but often reflect the quality of care delivered. To address this, we employ a **Long Short-Term Memory (LSTM)** neural network—a specialized type of Recurrent Neural Network (RNN) that excels at modeling sequential data over time. LSTM captures long-term dependencies and trends in historical hospital metrics, which helps in accurate forecasting of patient outcomes.

The model is trained on multiple time-dependent features, including:
- Number of patients admitted
- Average length of stay
- Average lab result score
- Hospital resource utilization

The data is normalized, fed into the LSTM, and used to predict the future readmission rate. The model generalizes patterns over time, offering predictive insights based on recent 30-day trends.

### 💡 Why It Matters:
- **Reduce avoidable readmissions**: Helps prevent unnecessary patient returns through early alerts.
- **Improve hospital efficiency**: Aids in planning staff, beds, and resource allocation.
- **Enhance patient care**: Enables targeted interventions for high-risk individuals.
- **Financial savings**: Avoiding readmissions reduces penalty costs and boosts operational margins.

### 📊 Use Cases:
- Healthcare analytics teams
- Hospital management systems
- Insurance providers assessing risk

---

## 📬 Contact
Feel free to reach out via [subhashg5397@gmail.com](mailto:subhashg5397@gmail.com) or open an issue in this repo.
