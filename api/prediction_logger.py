import os
import json
import sqlite3
from datetime import datetime

# PREDICTION LOGGER
# Logs every prediction to a SQLite database
class PredictionLogger:
    def __init__(self):
        self.db_path = "/app/logs/predictions.db"
        os.makedirs("/app/logs", exist_ok=True)
        self._init_db()

    def _init_db(self):
        """Create predictions table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS predictions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp        TEXT NOT NULL,
                store_id         INTEGER,
                input_features   TEXT,
                predicted_sales  REAL,
                inference_ms     REAL,
                model_version    TEXT
            )
        """)
        conn.commit()
        conn.close()

    def log(self, input_data: dict, predicted_sales: float,
            inference_ms: float, model_version: str):
        """Log a single prediction to the database."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO predictions
                (timestamp, store_id, input_features, predicted_sales, inference_ms, model_version)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            datetime.utcnow().isoformat(),
            input_data.get("Store"),
            json.dumps(input_data),
            predicted_sales,
            inference_ms,
            model_version,
        ))
        conn.commit()
        conn.close()

    def get_recent(self, limit: int = 100) -> list:
        """Retrieve recent predictions for monitoring."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            SELECT timestamp, store_id, predicted_sales, inference_ms, model_version
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
        """, (limit,))
        rows = cursor.fetchall()
        conn.close()
        return [
            {
                "timestamp":       r[0],
                "store_id":        r[1],
                "predicted_sales": r[2],
                "inference_ms":    r[3],
                "model_version":   r[4],
            }
            for r in rows
        ]

    def get_stats(self) -> dict:
        """Get aggregate stats for monitoring dashboard."""
        conn = sqlite3.connect(self.db_path)

        total = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        avg_latency = conn.execute("SELECT AVG(inference_ms) FROM predictions").fetchone()[0]
        p95_latency = conn.execute("""
            SELECT inference_ms FROM predictions
            ORDER BY inference_ms
            LIMIT 1 OFFSET (SELECT CAST(COUNT(*) * 0.95 AS INT) FROM predictions)
        """).fetchone()
        avg_sales = conn.execute("SELECT AVG(predicted_sales) FROM predictions").fetchone()[0]

        # Predictions per hour (last 24h)
        per_hour = conn.execute("""
            SELECT strftime('%Y-%m-%dT%H:00:00', timestamp) as hour, COUNT(*) as count
            FROM predictions
            WHERE timestamp >= datetime('now', '-24 hours')
            GROUP BY hour
            ORDER BY hour
        """).fetchall()

        conn.close()

        return {
            "total_predictions": total,
            "avg_latency_ms":    round(avg_latency, 2) if avg_latency else 0,
            "p95_latency_ms":    round(p95_latency[0], 2) if p95_latency else 0,
            "avg_predicted_sales": round(avg_sales, 2) if avg_sales else 0,
            "predictions_per_hour": [{"hour": r[0], "count": r[1]} for r in per_hour],
        }


prediction_logger = PredictionLogger()