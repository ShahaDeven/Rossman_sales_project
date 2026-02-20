"""
API Tests — runs against a test instance of the FastAPI app
using dummy model artifacts (no real model weights needed).
"""
import pytest
import pickle
import json
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from fastapi.testclient import TestClient


# ─────────────────────────────────────────────
# FIXTURES — Create dummy artifacts before tests
# ─────────────────────────────────────────────
@pytest.fixture(autouse=True, scope="session")
def create_dummy_artifacts(tmp_path_factory):
    """Create minimal dummy model artifacts so the API can load."""
    import os

    # Already created by CI script, but create here too for local runs
    if not os.path.exists("scaler.pkl"):
        scaler = MinMaxScaler()
        scaler.fit(np.zeros((10, 21)))
        with open("scaler.pkl", "wb") as f:
            pickle.dump(scaler, f)

    if not os.path.exists("features.json"):
        features = [
            "Store", "DayOfWeek", "Customers", "Open", "Promo",
            "StateHoliday", "SchoolHoliday", "StoreType", "Assortment",
            "CompetitionDistance", "Promo2", "Year", "Month", "Day",
            "WeekOfYear", "IsPromoMonth", "IsWeekend", "IsHoliday",
            "IsMonthStart", "IsMonthEnd", "DaysSincePromo"
        ]
        with open("features.json", "w") as f:
            json.dump(features, f)

    if not os.path.exists("model_state_dict.pth"):
        class _LSTM(nn.Module):
            def __init__(self):
                super().__init__()
                self.lstm1    = nn.LSTM(21, 256, batch_first=True)
                self.dropout1 = nn.Dropout(0.333)
                self.lstm2    = nn.LSTM(256, 128, batch_first=True)
                self.dropout2 = nn.Dropout(0.333)
                self.fc1      = nn.Linear(128, 32)
                self.fc2      = nn.Linear(32, 32)
                self.out      = nn.Linear(32, 1)
                self.relu     = nn.ReLU()
            def forward(self, x):
                out, _ = self.lstm1(x)
                out = self.dropout1(out)
                out, _ = self.lstm2(out)
                out = self.dropout2(out)
                out = out[:, -1, :]
                out = self.relu(self.fc1(out))
                out = self.relu(self.fc2(out))
                return self.out(out).squeeze(1)
        torch.save(_LSTM().state_dict(), "model_state_dict.pth")


@pytest.fixture(scope="session")
def client(create_dummy_artifacts):
    """Create a test client with the FastAPI app."""
    from api.main import app
    with TestClient(app) as c:
        yield c


# ─────────────────────────────────────────────
# SAMPLE PAYLOAD
# ─────────────────────────────────────────────
SAMPLE_PAYLOAD = {
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
    "DaysSincePromo": 3,
}


# ─────────────────────────────────────────────
# TESTS
# ─────────────────────────────────────────────
class TestHealth:
    def test_health_returns_200(self, client):
        response = client.get("/health")
        assert response.status_code == 200

    def test_health_response_schema(self, client):
        data = client.get("/health").json()
        assert "status" in data
        assert "model_version" in data
        assert "uptime_seconds" in data
        assert "timestamp" in data

    def test_health_status_is_ok(self, client):
        data = client.get("/health").json()
        assert data["status"] == "ok"


class TestModelInfo:
    def test_model_info_returns_200(self, client):
        response = client.get("/model-info")
        assert response.status_code == 200

    def test_model_info_has_features(self, client):
        data = client.get("/model-info").json()
        assert "features" in data
        assert len(data["features"]) == 21

    def test_model_info_has_metrics(self, client):
        data = client.get("/model-info").json()
        assert "metrics" in data


class TestPredict:
    def test_predict_returns_200(self, client):
        response = client.post("/predict", json=SAMPLE_PAYLOAD)
        assert response.status_code == 200

    def test_predict_response_schema(self, client):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert "predicted_sales" in data
        assert "model_version" in data
        assert "inference_latency_ms" in data
        assert "timestamp" in data

    def test_predicted_sales_is_positive(self, client):
        data = client.post("/predict", json=SAMPLE_PAYLOAD).json()
        assert data["predicted_sales"] > 0

    def test_predict_rejects_missing_field(self, client):
        bad_payload = SAMPLE_PAYLOAD.copy()
        del bad_payload["Store"]
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_rejects_invalid_day_of_week(self, client):
        bad_payload = SAMPLE_PAYLOAD.copy()
        bad_payload["DayOfWeek"] = 8  # invalid, must be 1-7
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_predict_rejects_negative_customers(self, client):
        bad_payload = SAMPLE_PAYLOAD.copy()
        bad_payload["Customers"] = -1
        response = client.post("/predict", json=bad_payload)
        assert response.status_code == 422


class TestMonitoring:
    def test_recent_predictions_returns_200(self, client):
        response = client.get("/predictions/recent")
        assert response.status_code == 200

    def test_prediction_stats_returns_200(self, client):
        response = client.get("/predictions/stats")
        assert response.status_code == 200

    def test_root_redirects_to_docs(self, client):
        response = client.get("/", follow_redirects=False)
        assert response.status_code in [301, 302, 307, 308]