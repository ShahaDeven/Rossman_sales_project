# 📈 Rossmann Store Sales Forecasting Agent

![Python](https://img.shields.io/badge/Python-3.9%2B-blue) 
![Deep%20Learning](https://img.shields.io/badge/Deep%20Learning-LSTM-red) 
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange) 
![Time%20Series](https://img.shields.io/badge/Time%20Series-Forecasting-green)

A production-style **time-series sales forecasting system** designed to predict daily retail sales for Rossmann stores using **LSTM-based deep learning models**.

This project enables accurate short-term and mid-term sales forecasting to support inventory planning, staffing, and operational decision-making.

Built on the **Rossmann Store Sales dataset**, this project demonstrates advanced **sequence modeling**, temporal feature engineering, hyperparameter tuning, and multi-horizon forecasting.

---

## 🚀 Key Features

### 🧠 Phase 1: Data Engineering & Preprocessing
- **Dataset Integration:** Merges `train`, `test`, and `store` datasets into a unified analytical table.
- **Outlier Handling:** Removes extreme sales and customer values using IQR-based filtering.
- **Temporal Feature Engineering:** Extracts year, month, day, and ISO week from date fields.
- **Categorical Encoding:** Encodes store type, assortment, and holiday indicators.
- **Scaling:** Applies MinMax scaling to prepare data for neural sequence models.

---

### 🤖 Phase 2: LSTM-Based Forecasting Model
- **Sequence Modeling:** Converts tabular data into rolling 30-day time windows.
- **Stacked LSTM Architecture:** Learns long-term temporal dependencies in sales patterns.
- **Regularization:** Dropout layers to reduce overfitting.
- **Early Stopping:** Prevents unnecessary training once validation loss plateaus.

---

### 📊 Phase 3: Evaluation & Forecasting
- **Robust Evaluation:** Assessed using MAE, RMSE, and R² metrics.
- **Next-Day Forecasting:** Generates single-step sales predictions.
- **Multi-Horizon Forecasting:** Produces 7-day and 30-day forward-looking forecasts.
- **Visual Diagnostics:** Actual vs predicted plots and forecast horizon visualizations.

---

### ⚡ Model Optimization
- **Hyperparameter Tuning:** Optimized LSTM units, dropout rate, dense layers, learning rate, batch size, and epochs.
- **Best Configuration Identified:** Balances predictive accuracy and generalization.

---

## 🛠️ Tech Stack
- **Programming Language:** Python 3.9+
- **Deep Learning:** TensorFlow / Keras (LSTM)
- **Machine Learning:** Scikit-learn
- **Data Processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Environment:** Jupyter Notebook

---

## 📊 Model Performance

### Best Model Evaluation
```text
MAE  : 945.18
RMSE : 1415.43
R²   : 0.8191
```

### Best Hyperparameters
```json
{
  "lstm_units": 256,
  "dropout_rate": 0.333095499066527,
  "dense_units_1": 32,
  "dense_units_2": 32,
  "learning_rate": 0.0008919224878354329,
  "batch_size": 32,
  "epochs": 11
}
```

## 📂 Project Structure
```bash
Rossmann_sales_project/
├── sales.py
├── sales.ipynb
├── train.csv
├── test.csv
├── store.csv
├── requirements.txt
└── README.md
```

## ⚡ Installation & Setup
1. Clone the Repository
```bash
git clone https://github.com/ShahaDeven/Rossman_sales_project.git
cd Rossman_sales_project
```

2. Create Virtual Environment
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 📝 Future Roadmap
- Walk-forward validation for time-series robustness
- Store-level or cluster-specific forecasting
- Probabilistic forecasting with confidence intervals
- Streamlit-based interactive forecasting dashboard
