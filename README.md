# 🧠 Time Series Sales Forecasting with LSTM

This project implements a deep learning-based time series forecasting model to predict retail store sales using historical data. It leverages stacked LSTM neural networks and autoregressive forecasting to accurately estimate sales for the next 1, 7, and 30 days.

---

## 📊 Overview

- Dataset: Rossmann Store Sales (from Kaggle)
- Model: Bidirectional + stacked LSTM with dropout and regularization
- Objective: Forecast daily sales for each store using temporal and contextual features
- Output: Daily sales predictions for future time horizons (1, 7, 30 days)
- Evaluation Metrics: MAE, RMSE, R² Score
- Optimization: Hyperparameter tuning with Optuna

---

## 🔧 Features & Techniques

✔ Cleaned and preprocessed sales data, removed outliers  
✔ Engineered date-based features (DayOfWeek, IsWeekend, WeekOfYear, etc.)  
✔ Scaled inputs and target using MinMaxScaler  
✔ Generated input sequences for LSTM with a window of 30 days  
✔ Implemented:
  - Next-day sales forecast  
  - 7-day rolling forecast  
  - 30-day rolling forecast  
✔ Tuned LSTM architecture using Optuna  
✔ Visualized predictions using both smoothed and downsampled plots 
