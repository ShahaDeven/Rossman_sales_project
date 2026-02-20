import time
import numpy as np
import torch
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from api.schemas import (
    PredictionRequest,
    PredictionResponse,
    HealthResponse,
    ModelInfoResponse,
)
from api.model_loader import model_loader
from api.prediction_logger import prediction_logger
from sklearn.preprocessing import LabelEncoder
from fastapi.responses import RedirectResponse

# APP STARTUP / SHUTDOWN
START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load model on startup
    print("Starting up — loading model...")
    model_loader.load()
    print("Model ready. API is live.")
    yield
    # Cleanup on shutdown
    print("Shutting down.")

app = FastAPI(
    title="Rossmann Sales Forecasting API",
    description="Production ML API for predicting Rossmann store daily sales using an LSTM model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")

# HELPER — preprocess one request into model input
def preprocess_request(request: PredictionRequest) -> torch.Tensor:
    """
    Convert a PredictionRequest into a (1, seq_len, n_features) tensor.
    Since the API receives a single row (not a sequence), we replicate it
    across the sequence length. This is a valid approach for inference
    when we don't have a full historical window available.
    """
    
    feature_values = []
    input_dict = request.model_dump()

    enc = LabelEncoder()

    store_type_map  = {"a": 0, "b": 1, "c": 2, "d": 3}
    assortment_map  = {"a": 0, "b": 1, "c": 2}
    holiday_map     = {"0": 0, "a": 1, "b": 2, "c": 3}

    input_dict["StoreType"]    = store_type_map.get(input_dict["StoreType"].lower(), 0)
    input_dict["Assortment"]   = assortment_map.get(input_dict["Assortment"].lower(), 0)
    input_dict["StateHoliday"] = holiday_map.get(str(input_dict["StateHoliday"]).lower(), 0)

    for feat in model_loader.features:
        feature_values.append(float(input_dict.get(feat, 0)))

    row = np.array([feature_values], dtype=np.float32)
    row_scaled = model_loader.scaler.transform(row)

    seq_length = 30
    sequence = np.tile(row_scaled, (seq_length, 1))         
    tensor   = torch.tensor(sequence).unsqueeze(0)          

    return tensor


# ENDPOINTS
@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    """Returns model status, version, and uptime."""
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return HealthResponse(
        status="ok",
        model_version=str(model_loader.model_version),
        uptime_seconds=round(time.time() - START_TIME, 2),
        timestamp=datetime.utcnow().isoformat(),
    )


@app.get("/model-info", response_model=ModelInfoResponse, tags=["Monitoring"])
async def model_info():
    """Returns metadata about the currently loaded model from MLflow registry."""
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    return ModelInfoResponse(
        model_name="rossmann_lstm",
        model_version=str(model_loader.model_version),
        run_id=model_loader.run_id,
        features=model_loader.features,
        metrics=model_loader.metrics,
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(request: PredictionRequest):
    """
    Accepts store features and returns a sales forecast.
    Logs every prediction to the database for monitoring.
    """
    if not model_loader.is_ready():
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        tensor = preprocess_request(request)

        start = time.time()
        model_loader.model.eval()
        with torch.no_grad():
            tensor = tensor.to(model_loader.device)
            log_pred = model_loader.model(tensor).cpu().item()

        inference_ms = round((time.time() - start) * 1000, 2)

        predicted_sales = round(float(np.expm1(log_pred)), 2)

        prediction_logger.log(
            input_data=request.model_dump(),
            predicted_sales=predicted_sales,
            inference_ms=inference_ms,
            model_version=str(model_loader.model_version),
        )

        return PredictionResponse(
            predicted_sales=predicted_sales,
            model_version=str(model_loader.model_version),
            inference_latency_ms=inference_ms,
            timestamp=datetime.utcnow().isoformat(),
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/predictions/recent", tags=["Monitoring"])
async def recent_predictions(limit: int = 50):
    """Returns the most recent predictions logged."""
    return prediction_logger.get_recent(limit=limit)


@app.get("/predictions/stats", tags=["Monitoring"])
async def prediction_stats():
    """Returns aggregate prediction stats for the monitoring dashboard."""
    return prediction_logger.get_stats()