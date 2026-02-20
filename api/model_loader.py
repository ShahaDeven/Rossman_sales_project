import os
import json
import pickle
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch

# LSTM MODEL CLASS (must match train.py exactly)
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden1=256, hidden2=128,
                 dropout=0.333, dense1=32, dense2=32):
        super(LSTMModel, self).__init__()

        self.lstm1    = nn.LSTM(input_size=input_size, hidden_size=hidden1,
                                num_layers=1, batch_first=True)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2    = nn.LSTM(input_size=hidden1, hidden_size=hidden2,
                                num_layers=1, batch_first=True)
        self.dropout2 = nn.Dropout(dropout)
        self.fc1      = nn.Linear(hidden2, dense1)
        self.fc2      = nn.Linear(dense1, dense2)
        self.out      = nn.Linear(dense2, 1)
        self.relu     = nn.ReLU()

    def forward(self, x):
        out, _ = self.lstm1(x)
        out = self.dropout1(out)
        out, _ = self.lstm2(out)
        out = self.dropout2(out)
        out = out[:, -1, :]
        out = self.relu(self.fc1(out))
        out = self.relu(self.fc2(out))
        out = self.out(out)
        return out.squeeze(1)


# MODEL LOADER
class ModelLoader:
    def __init__(self):
        self.model         = None
        self.scaler        = None
        self.features      = None
        self.model_version = "1"
        self.run_id        = "unknown"
        self.metrics       = {}
        self.device        = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.root_dir      = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        self.scaler_path   = os.path.join(self.root_dir, "scaler.pkl")
        self.features_path = os.path.join(self.root_dir, "features.json")
        self.mlruns_path   = os.path.join(self.root_dir, "mlruns")
        self.db_path       = os.path.join(self.root_dir, "mlflow.db")

    def _find_model_pth(self) -> str:
        plain_path = os.path.join(self.root_dir, "model_state_dict.pth")
        if os.path.exists(plain_path):
            return plain_path
        raise FileNotFoundError("model_state_dict.pth not found in project root.")

    def load(self):
        """Load model weights directly from local mlruns/ folder."""
        print("Loading model from local mlruns/...")

        # ── Load features list first to get input size ──
        with open(self.features_path, "r") as f:
            self.features = json.load(f)

        input_size = len(self.features)
        print(f"  Input size: {input_size} features")

        # ── Build model with same architecture as training ──
        self.model = LSTMModel(
            input_size=input_size,
            hidden1=256,
            hidden2=128,
            dropout=0.333,
            dense1=32,
            dense2=32,
        ).to(self.device)

        # ── Find and load model weights ──
        model_pth = self._find_model_pth()
        print(f"  Found weights at: {model_pth}")

        state_dict = torch.load(model_pth, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

        # ── Load scaler ──
        with open(self.scaler_path, "rb") as f:
            self.scaler = pickle.load(f)

        # ── Try to get metrics from mlflow.db ──
        try:
            mlflow.set_tracking_uri(f"sqlite:///{self.db_path}")
            client = mlflow.tracking.MlflowClient()
            versions = client.get_latest_versions("rossmann_lstm")
            if versions:
                latest        = versions[0]
                self.model_version = latest.version
                self.run_id   = latest.run_id
                run           = client.get_run(self.run_id)
                self.metrics  = {
                    "test_mae":  run.data.metrics.get("test_mae"),
                    "test_rmse": run.data.metrics.get("test_rmse"),
                    "test_r2":   run.data.metrics.get("test_r2"),
                }
        except Exception as e:
            print(f"  Could not load MLflow metadata: {e}")
            self.metrics = {
                "test_mae":  534.61,
                "test_rmse": 779.68,
                "test_r2":   0.9451,
            }

        print(f"Model ready — version: {self.model_version} | device: {self.device}")
        return self

    def is_ready(self) -> bool:
        return all([
            self.model is not None,
            self.scaler is not None,
            self.features is not None,
        ])


model_loader = ModelLoader()