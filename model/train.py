import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import mlflow
import mlflow.pytorch
from mlflow.models.signature import infer_signature
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# LOAD CONFIG
CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


# 1. LOAD DATA
def load_data(cfg):
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    train_df = pd.read_csv(os.path.join(data_dir, cfg["data"]["train_file"]), low_memory=False)
    test_df  = pd.read_csv(os.path.join(data_dir, cfg["data"]["test_file"]),  low_memory=False)
    store_df = pd.read_csv(os.path.join(data_dir, cfg["data"]["store_file"]), low_memory=False)

    merged_df      = train_df.merge(store_df, how="left", on="Store")
    merged_test_df = test_df.merge(store_df,  how="left", on="Store")

    merged_df["Date"]      = pd.to_datetime(merged_df["Date"])
    merged_test_df["Date"] = pd.to_datetime(merged_test_df["Date"])

    print(f"Loaded {len(merged_df):,} training rows")
    return merged_df, merged_test_df


# 2. PREPROCESS
def remove_outliers(df, col):
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR    = Q3 - Q1
    before = len(df)
    df = df[(df[col] >= Q1 - 1.5 * IQR) & (df[col] <= Q3 + 1.5 * IQR)]
    print(f"  Removed {before - len(df):,} outliers from '{col}'")
    return df


def split_date(df):
    df["Year"]       = df["Date"].dt.year
    df["Month"]      = df["Date"].dt.month
    df["Day"]        = df["Date"].dt.day
    df["WeekOfYear"] = df["Date"].dt.isocalendar().week.astype(int)


MONTH_MAP = {
    "Jan":1,"Feb":2,"Mar":3,"Apr":4,"May":5,"Jun":6,
    "Jul":7,"Aug":8,"Sept":9,"Oct":10,"Nov":11,"Dec":12
}

def convert_promo_interval(promo_str):
    if pd.isna(promo_str):
        return ""
    return ",".join(str(MONTH_MAP[m]) for m in promo_str.split(","))

def check_promo_month(row):
    if not row["PromoInterval_converted"]:
        return 0
    promo_months = list(map(int, row["PromoInterval_converted"].split(",")))
    return 1 if row["Month"] in promo_months else 0


def add_new_features(df):
    df["IsWeekend"]    = df["DayOfWeek"].isin([6, 7]).astype(int)
    df["IsHoliday"]    = (
        (df["SchoolHoliday"] == 1) |
        (df["StateHoliday"].astype(str) != "0")
    ).astype(int)
    df["IsMonthStart"] = (df["Day"] <= 3).astype(int)
    df["IsMonthEnd"]   = (df["Day"] >= 28).astype(int)

    df = df.sort_values(["Store", "Date"])
    df["PromoStart"] = df.groupby("Store")["Promo"].transform(
        lambda x: x.ne(x.shift()).cumsum()
    )
    df["DaysSincePromo"] = df.groupby(["Store", "PromoStart"]).cumcount()
    df = df.drop(columns=["PromoStart"])

    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(
        df["CompetitionDistance"].median()
    )
    return df


def preprocess(merged_df):
    df = merged_df.copy()

    print("\nRemoving outliers...")
    df = remove_outliers(df, "Sales")
    df = remove_outliers(df, "Customers")

    print("Engineering date features...")
    split_date(df)

    print("Engineering promo features...")
    df["PromoInterval_converted"] = df["PromoInterval"].apply(convert_promo_interval)
    df["IsPromoMonth"] = df.apply(check_promo_month, axis=1)

    print("Adding new features (IsWeekend, IsHoliday, MonthBounds, DaysSincePromo)...")
    df = add_new_features(df)

    drop_cols = [
        "PromoInterval", "PromoInterval_converted",
        "CompetitionOpenSinceMonth", "CompetitionOpenSinceYear",
        "Promo2SinceWeek", "Promo2SinceYear",
    ]
    df = df.drop(columns=drop_cols)

    encoder = LabelEncoder()
    df["StoreType"]    = encoder.fit_transform(df["StoreType"])
    df["Assortment"]   = encoder.fit_transform(df["Assortment"])
    df["StateHoliday"] = encoder.fit_transform(df["StateHoliday"].astype(str))

    df = df.sort_values(by="Date").reset_index(drop=True)

    print(f"Preprocessing done. Shape: {df.shape}")
    return df


# 3. LOG-TRANSFORM + SCALE + SEQUENCES
def scale_and_sequence(df, seq_length=30):
    cache_dir    = os.path.join(os.path.dirname(__file__), "data", "cache")
    cache_X      = os.path.join(cache_dir, f"X_seq{seq_length}.npy")
    cache_y      = os.path.join(cache_dir, f"y_seq{seq_length}.npy")
    cache_meta   = os.path.join(cache_dir, f"meta_seq{seq_length}.json")
    cache_scaler = os.path.join(cache_dir, "scaler.pkl")

    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_X) and os.path.exists(cache_y):
        print("Cache found — loading sequences from disk (skipping rebuild)...")
        X = np.load(cache_X)
        y = np.load(cache_y)
        with open(cache_meta) as f:
            meta = json.load(f)
        features = meta["features"]
        with open(cache_scaler, "rb") as f:
            scaler = pickle.load(f)
        print(f"Loaded {len(X):,} sequences of length {seq_length} from cache")
        return X.astype(np.float32), y.astype(np.float32), scaler, features

    print("No cache found — building sequences (this takes a while, but only happens once)...")
    features = df.drop(["Sales", "Date"], axis=1).columns.tolist()

    df["Sales_log"] = np.log1p(df["Sales"])

    scaler = MinMaxScaler()
    df[features] = scaler.fit_transform(df[features])

    X, y  = [], []
    total = len(df) - seq_length
    for i in range(total):
        if i % 100_000 == 0:
            print(f"  Building sequences... {i:,}/{total:,}")
        X.append(df.iloc[i : i + seq_length][features].values)
        y.append(df.iloc[i + seq_length - 1]["Sales_log"])

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    print("Saving sequences to cache...")
    np.save(cache_X, X)
    np.save(cache_y, y)
    with open(cache_meta, "w") as f:
        json.dump({"features": features, "seq_length": seq_length}, f)
    with open(cache_scaler, "wb") as f:
        pickle.dump(scaler, f)

    print(f"Created and cached {len(X):,} sequences of length {seq_length}")
    return X, y, scaler, features


# 4. PYTORCH LSTM MODEL
class LSTMModel(nn.Module):
    def __init__(self, input_size, cfg):
        super(LSTMModel, self).__init__()
        m = cfg["model"]

        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=m["lstm_units_1"],
            num_layers=1,
            batch_first=True,
        )
        self.dropout1 = nn.Dropout(m["dropout_rate"])

        self.lstm2 = nn.LSTM(
            input_size=m["lstm_units_1"],
            hidden_size=m["lstm_units_2"],
            num_layers=1,
            batch_first=True,
        )
        self.dropout2 = nn.Dropout(m["dropout_rate"])

        self.fc1  = nn.Linear(m["lstm_units_2"], m["dense_units_1"])
        self.fc2  = nn.Linear(m["dense_units_1"], m["dense_units_2"])
        self.out  = nn.Linear(m["dense_units_2"], 1)
        self.relu = nn.ReLU()

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


# 5. TRAIN ONE EPOCH
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, total_mae = 0.0, 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        preds = model(X_batch)
        loss  = criterion(preds, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_mae  += torch.mean(torch.abs(preds - y_batch)).item()
    n = len(loader)
    return total_loss / n, total_mae / n


# 6. VALIDATE ONE EPOCH
def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_mae = 0.0, 0.0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            preds = model(X_batch)
            loss  = criterion(preds, y_batch)
            total_loss += loss.item()
            total_mae  += torch.mean(torch.abs(preds - y_batch)).item()
    n = len(loader)
    return total_loss / n, total_mae / n


# 7. EVALUATE — batched to avoid VRAM OOM
def evaluate(model, X_test_t, y_test, device, batch_size=512):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(X_test_t), batch_size):
            batch = X_test_t[i : i + batch_size].to(device)
            out   = model(batch).cpu().numpy()
            preds.append(out)
            torch.cuda.empty_cache()

    y_pred_log  = np.concatenate(preds)
    y_pred_real = np.expm1(y_pred_log)
    y_test_real = np.expm1(y_test)

    mae  = mean_absolute_error(y_test_real, y_pred_real)
    rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
    r2   = r2_score(y_test_real, y_pred_real)

    return mae, rmse, r2, y_pred_real, y_test_real


# 8. SAVE ARTIFACTS
def save_artifacts(train_losses, val_losses, y_test_real, y_pred_real, scaler, features):
    plt.figure(figsize=(10, 4))
    plt.plot(train_losses, label="Train Loss")
    plt.plot(val_losses,   label="Val Loss")
    plt.title("Training vs Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig("loss_curve.png")
    mlflow.log_artifact("loss_curve.png")
    plt.close()

    plt.figure(figsize=(12, 5))
    plt.plot(y_test_real[:300], label="Actual",    alpha=0.8)
    plt.plot(y_pred_real[:300], label="Predicted", alpha=0.8)
    plt.title("Actual vs Predicted Sales (first 300 samples)")
    plt.xlabel("Sample")
    plt.ylabel("Sales")
    plt.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted.png")
    mlflow.log_artifact("actual_vs_predicted.png")
    plt.close()

    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact("scaler.pkl")

    with open("features.json", "w") as f:
        json.dump(features, f)
    mlflow.log_artifact("features.json")


# 9. MAIN TRAIN FUNCTION
def train(run_name="lstm_rossmann_pytorch"):
    cfg = load_config()
    m   = cfg["model"]

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    print("MLflow tracking URI: sqlite:///mlflow.db")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("\nLoading data...")
    merged_df, _ = load_data(cfg)

    print("\nPreprocessing...")
    df = preprocess(merged_df)

    print("\nScaling and creating sequences...")
    X, y, scaler, features = scale_and_sequence(df, seq_length=m["seq_length"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=m["test_size"], shuffle=False
    )
    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Features ({len(features)}): {features}")

    X_train_t = torch.tensor(X_train)
    y_train_t = torch.tensor(y_train)
    X_test_t  = torch.tensor(X_test)
    y_test_t  = torch.tensor(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=m["batch_size"],
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=m["batch_size"],
        shuffle=False,
    )

    # ── MLflow experiment ──
    mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\nMLflow Run ID: {run.info.run_id}")

        mlflow.log_params(m)
        mlflow.log_param("framework",     "pytorch")
        mlflow.log_param("device",        str(device))
        mlflow.log_param("num_features",  len(features))
        mlflow.log_param("features",      str(features))
        mlflow.log_param("train_samples", len(X_train))
        mlflow.log_param("test_samples",  len(X_test))
        mlflow.log_param("log_transform", True)

        model     = LSTMModel(input_size=X_train.shape[2], cfg=cfg).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=m["learning_rate"])
        criterion = nn.MSELoss()

        print(f"\nModel architecture:\n{model}\n")

        best_val_loss  = float("inf")
        patience_count = 0
        patience       = 5
        best_weights   = None
        train_losses, val_losses = [], []

        print("Training...")
        for epoch in range(1, m["epochs"] + 1):
            tr_loss, tr_mae = train_epoch(model, train_loader, optimizer, criterion, device)
            vl_loss, vl_mae = val_epoch(model, val_loader, criterion, device)

            train_losses.append(tr_loss)
            val_losses.append(vl_loss)

            mlflow.log_metrics({
                "train_loss": tr_loss,
                "train_mae":  tr_mae,
                "val_loss":   vl_loss,
                "val_mae":    vl_mae,
            }, step=epoch)

            print(
                f"Epoch {epoch:3d}/{m['epochs']} | "
                f"Train Loss: {tr_loss:.5f} | Val Loss: {vl_loss:.5f} | "
                f"Train MAE: {tr_mae:.5f} | Val MAE: {vl_mae:.5f}"
            )

            if vl_loss < best_val_loss:
                best_val_loss  = vl_loss
                best_weights   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"\nEarly stopping triggered at epoch {epoch} (patience={patience})")
                    break

        model.load_state_dict(best_weights)

        torch.cuda.empty_cache()
        mae, rmse, r2, y_pred_real, y_test_real = evaluate(model, X_test_t, y_test, device)

        print(f"\n{'─'*40}")
        print(f"Test MAE:  {mae:,.2f}")
        print(f"Test RMSE: {rmse:,.2f}")
        print(f"Test R²:   {r2:.4f}")
        print(f"{'─'*40}\n")

        mlflow.log_metrics({
            "test_mae":  mae,
            "test_rmse": rmse,
            "test_r2":   r2,
        })

        save_artifacts(train_losses, val_losses, y_test_real, y_pred_real, scaler, features)

        sample_input = X_train[:5]
        model.eval()
        with torch.no_grad():
            sample_output = model(torch.tensor(sample_input).to(device)).cpu().numpy()
            torch.cuda.empty_cache()

        signature = infer_signature(sample_input, sample_output)
        mlflow.pytorch.log_model(
            model,
            artifact_path="lstm_model",
            registered_model_name=cfg["mlflow"]["model_name"],
            signature=signature,
        )

        print(f"Model registered as '{cfg['mlflow']['model_name']}'")
        print(f"Done! Run: mlflow ui  →  http://localhost:5000")

    return model, scaler, features

# ENTRY POINT
if __name__ == "__main__":
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available:  {torch.cuda.is_available()}")
    train()