from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

# INPUT SCHEMA
class PredictionRequest(BaseModel):
    """
    Input features for a single sales prediction.
    Matches the 21 features the model was trained on.
    """
    Store: int                  = Field(..., ge=1, description="Store ID")
    DayOfWeek: int              = Field(..., ge=1, le=7, description="Day of week (1=Mon, 7=Sun)")
    Customers: int              = Field(..., ge=0, description="Number of customers")
    Open: int                   = Field(..., ge=0, le=1, description="1=open, 0=closed")
    Promo: int                  = Field(..., ge=0, le=1, description="1=promo running")
    StateHoliday: str           = Field(..., description="0=none, a=public, b=easter, c=christmas")
    SchoolHoliday: int          = Field(..., ge=0, le=1, description="1=school holiday")
    StoreType: str              = Field(..., description="Store type: a, b, c, or d")
    Assortment: str             = Field(..., description="Assortment level: a, b, or c")
    CompetitionDistance: float  = Field(..., ge=0, description="Distance to nearest competitor (m)")
    Promo2: int                 = Field(..., ge=0, le=1, description="1=store participates in Promo2")
    Year: int                   = Field(..., ge=2000, le=2100)
    Month: int                  = Field(..., ge=1, le=12)
    Day: int                    = Field(..., ge=1, le=31)
    WeekOfYear: int             = Field(..., ge=1, le=53)
    IsPromoMonth: int           = Field(..., ge=0, le=1)
    IsWeekend: int              = Field(..., ge=0, le=1)
    IsHoliday: int              = Field(..., ge=0, le=1)
    IsMonthStart: int           = Field(..., ge=0, le=1)
    IsMonthEnd: int             = Field(..., ge=0, le=1)
    DaysSincePromo: int         = Field(..., ge=0)

    class Config:
        json_schema_extra = {
            "example": {
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
            }
        }


# OUTPUT SCHEMA
class PredictionResponse(BaseModel):
    predicted_sales: float      = Field(..., description="Predicted sales in original scale")
    model_version: str          = Field(..., description="MLflow model version used")
    inference_latency_ms: float = Field(..., description="Inference time in milliseconds")
    timestamp: str              = Field(..., description="Prediction timestamp (ISO 8601)")


# HEALTH SCHEMA
class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float
    timestamp: str


# MODEL INFO SCHEMA
class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    run_id: str
    features: List[str]
    metrics: dict
    timestamp: str