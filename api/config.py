"""Configuration settings for Fire Risk Predictor API"""

import os
from typing import Dict

# MinIO Storage Configuration (Railway internal network)
MINIO_BASE_URL = os.getenv("MINIO_BASE_URL", "http://bucket.railway.internal:9000")
MINIO_BUCKET = "fire-risk-predictor"

MODEL_URLS: Dict[str, str] = {
    "RF": f"{MINIO_BASE_URL}/{MINIO_BUCKET}/RF.pkl",
    "MLP": f"{MINIO_BASE_URL}/{MINIO_BUCKET}/MLP.pkl",
    "XGBoost": f"{MINIO_BASE_URL}/{MINIO_BUCKET}/XGBoost.pkl",
}

DATASET_URL = f"{MINIO_BASE_URL}/{MINIO_BUCKET}/sisam_focos_2003.csv"

# Model metadata (optimal thresholds from notebook analysis)
# These will be used as defaults - can be overridden at prediction time
MODEL_METADATA: Dict[str, Dict] = {
    "RF": {
        "name": "Random Forest Regressor",
        "default_threshold": 0.5,
        "description": "Ensemble tree-based model",
    },
    "MLP": {
        "name": "Multi-Layer Perceptron",
        "default_threshold": 0.5,
        "description": "Neural network with StandardScaler preprocessing",
    },
    "XGBoost": {
        "name": "XGBoost Regressor",
        "default_threshold": 0.5,
        "description": "Gradient boosting model",
    },
}

# Columns to drop during preprocessing (from notebook analysis)
COLUMNS_TO_DROP_TRAINING = [
    "data_pas",
    "satelite",
    "pais",
    "estado",
    "municipio",
    "bioma",
    "numero_dias_sem_chuva",
    "precipitacao",
    "risco_fogo",
    "id_area_industrial",
    "frp",
    "time_diff_hours",
]

COLUMNS_TO_DROP_INFERENCE = COLUMNS_TO_DROP_TRAINING + [
    "datahora",
    "longitude",
    "latitude",
]

# Optional columns that will be dropped if present
OPTIONAL_COLUMNS_TO_DROP = ["Cluster"]

# Target column
TARGET_COLUMN = "incendio"

# API Settings
MAX_FILE_SIZE_MB = 100
CHUNK_SIZE = 10000  # For streaming large CSV files

# Cache settings
MODEL_CACHE_TTL_SECONDS = 3600  # 1 hour
