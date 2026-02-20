# Rossmann Sales Forecasting — PyTorch LSTM

Production ML pipeline for predicting Rossmann store daily sales using a PyTorch LSTM model, served via a REST API with a real-time monitoring dashboard.

![Python](https://img.shields.io/badge/Python-3.10-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-LSTM-EE4C2C) ![FastAPI](https://img.shields.io/badge/FastAPI-REST_API-009688) ![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b) ![MLflow](https://img.shields.io/badge/MLflow-Tracking-0194E2) ![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED) ![CI/CD](https://img.shields.io/badge/GitHub_Actions-CI%2FCD-2088FF)

---

## Overview

This project builds an end-to-end machine learning pipeline for the [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) dataset. It covers everything from model training to production serving.

| Metric | Value |
|---|---|
| Test MAE | 543.25 |
| Test RMSE | 787.83 |
| Test R² | 0.9440 |
| Inference Latency | ~7ms (GPU) |

---

## Project Structure

```
Roseman_sales/
├── api/                        # FastAPI serving layer
│   ├── main.py                 # API endpoints
│   ├── schemas.py              # Pydantic request/response models
│   ├── model_loader.py         # Model loading from MLflow
│   └── prediction_logger.py   # Prediction logging to SQLite
├── model/                      # Training code
│   ├── train.py                # LSTM training script
│   └── config.yaml             # Hyperparameters and settings
├── monitoring/                 # Streamlit dashboard
│   └── dashboard.py            # Monitoring + Try It Out UI
├── tests/                      # Pytest test suite
│   └── test_api.py
├── .github/workflows/
│   └── ci.yml                  # GitHub Actions CI/CD
├── Dockerfile                  # API container
├── Dockerfile.monitoring       # Dashboard container
├── docker-compose.yml          # Run both services together
├── mlflow.db                   # MLflow experiment tracking
├── scaler.pkl                  # Feature scaler
├── features.json               # Feature names list
└── model_state_dict.pth        # Trained model weights
```

---

## Architecture

```
                    ┌─────────────────────┐
                    │  Streamlit Dashboard│
                    │  localhost:8501     │
                    └────────┬────────────┘
                             │ HTTP
                    ┌────────▼────────────┐
                    │   FastAPI REST API  │
                    │   localhost:8000    │
                    └────────┬────────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
    ┌─────────▼──────┐           ┌──────────▼──────┐
    │  PyTorch LSTM  │           │   MLflow DB     │
    │  model weights │           │   mlflow.db     │
    └────────────────┘           └─────────────────┘
```

---

## Model

The LSTM model is trained on 21 engineered features including store type, promotions, competition distance, and calendar features.

**Architecture:**
- LSTM layer 1 → 256 hidden units
- Dropout (0.333)
- LSTM layer 2 → 128 hidden units
- Dropout (0.333)
- Dense → 32 → 32 → 1
- Log-transform on target (Sales), sequence length of 30

**Features:**
`Store`, `DayOfWeek`, `Customers`, `Open`, `Promo`, `StateHoliday`, `SchoolHoliday`, `StoreType`, `Assortment`, `CompetitionDistance`, `Promo2`, `Year`, `Month`, `Day`, `WeekOfYear`, `IsPromoMonth`, `IsWeekend`, `IsHoliday`, `IsMonthStart`, `IsMonthEnd`, `DaysSincePromo`

---

## Quickstart

### Option 1 — Docker (recommended)

```bash
docker-compose up --build
```

- API docs → http://localhost:8000/docs
- Dashboard → http://localhost:8501

### Option 2 — Local

**Install dependencies:**
```bash
pip install -r requirements.txt
```

**Start the API:**
```bash
uvicorn api.main:app --reload --port 8000
```

**Start the dashboard** (separate terminal):
```bash
streamlit run monitoring/dashboard.py
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | API status, model version, uptime |
| `GET` | `/model-info` | Model metadata and test metrics |
| `POST` | `/predict` | Predict daily sales for a store |
| `GET` | `/predictions/recent` | Last N logged predictions |
| `GET` | `/predictions/stats` | Aggregate stats for monitoring |

**Example prediction request:**

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Store": 1,
    "DayOfWeek": 5,
    "Customers": 555,
    "Open": 1,
    "Promo": 1,
    "StateHoliday": "0",
    "SchoolHoliday": 1,
    "StoreType": "c",
    "Assortment": "a",
    "CompetitionDistance": 1270.0,
    "Promo2": 0,
    "Year": 2015,
    "Month": 7,
    "Day": 31,
    "WeekOfYear": 31,
    "IsPromoMonth": 0,
    "IsWeekend": 0,
    "IsHoliday": 1,
    "IsMonthStart": 0,
    "IsMonthEnd": 1,
    "DaysSincePromo": 3
  }'
```

**Response:**
```json
{
  "predicted_sales": 5351.96,
  "model_version": "1",
  "inference_latency_ms": 7.2,
  "timestamp": "2026-02-20T15:00:00"
}
```

---

## Monitoring Dashboard

The Streamlit dashboard at `http://localhost:8501` has two tabs:

**📊 Monitoring**
- API status, model version, uptime
- Average and P95 inference latency
- Predictions per hour (last 24h)
- Latency distribution histogram
- Predicted sales over time
- Recent predictions table

**🧪 Try It Out**
- 5 preset scenarios (Busy Friday, Holiday Weekend, Christmas Eve, etc.)
- Interactive sliders and dropdowns for all 21 features
- Auto-calculated fields (WeekOfYear, IsWeekend, IsHoliday, etc.)
- Live prediction results with sales, latency, and model version

---

## Training

Training requires the Rossmann dataset files (`train.csv`, `test.csv`, `store.csv`) placed in `model/data/`.

```bash
python model/train.py
```

Experiment metrics, parameters, and model artifacts are tracked in MLflow:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

---

## CI/CD

GitHub Actions pipeline runs on every push:

- **Tests** — 13 pytest tests covering all API endpoints and input validation
- **Docker Build** — builds both images on push to `main`

Tests use dummy model artifacts so no real weights are needed in CI.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch LSTM |
| Experiment Tracking | MLflow |
| API | FastAPI + Uvicorn |
| Validation | Pydantic |
| Dashboard | Streamlit + Plotly |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Prediction Logging | SQLite |

---

## Challenges & Solutions

**Cross-platform MLflow artifact paths**
MLflow stores absolute file paths in its database when saving model artifacts. Transferring `mlflow.db` from a Linux server to Windows caused `OSError` on loading because the paths pointed to the server's filesystem. Solved by saving model weights as a plain PyTorch state dict (`model_state_dict.pth`) separately from MLflow, bypassing the artifact path issue entirely.

**CloudPickle version mismatch**
The model saved on the server (Linux, Python 3.12) could not be loaded locally (Windows, Python 3.10) due to incompatible CloudPickle versions used internally by MLflow. The fix was the same — saving a plain state dict instead of relying on MLflow's pickle-based model format.

**CUDA warm-up latency**
First inference request took ~1800ms due to CUDA kernel compilation. Subsequent requests dropped to ~7ms. Documented this behaviour in the dashboard so users are not alarmed by the first request being slow.

**Docker inter-container networking**
The Streamlit dashboard showed API as "Offline" inside Docker because `http://127.0.0.1:8000` resolves to the dashboard container itself, not the API container. Fixed by using the Docker service name `http://rossmann_api:8000` for container-to-container communication.

**CI/CD path and permission errors**
GitHub Actions runs in a clean environment with no `/app` directory, causing `PermissionError` when `prediction_logger.py` tried to create `/app/logs`. Fixed by using relative paths derived from `__file__` instead of hardcoded Docker paths, making the code environment-agnostic.

## Requirements

- Python 3.10
- CUDA-compatible GPU (optional, CPU fallback supported)
- Docker Desktop (for containerized setup)
