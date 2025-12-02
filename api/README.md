# 🔥 Fire Risk Predictor API

FastAPI backend for ML-based fire risk prediction using pre-trained models (Random Forest, MLP, XGBoost).

## Features

- **Single & Batch Predictions**: Predict fire risk for individual records or CSV files
- **Multiple Models**: Choose between Random Forest, MLP, or XGBoost
- **Threshold Optimization**: Automatically find optimal classification threshold
- **Model Caching**: Efficient in-memory caching with TTL
- **Evaluation Metrics**: Automatic metrics computation when ground truth is provided
- **CORS Enabled**: Ready for frontend integration

## Quick Start

### Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with hypercorn (production-ready)
hypercorn main:app --reload

# Or run with uvicorn (alternative)
uvicorn main:app --reload
```

The API will be available at `http://localhost:8000`

### Docker Deployment

```bash
# Build image
docker build -t fire-risk-api .

# Run container
docker run -p 8000:8000 fire-risk-api
```

### Railway Deployment

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template/-NvLj4?referralCode=CRJ8FE)

The `railway.json` is configured for automatic deployment.

## API Endpoints

### Core Endpoints

- `GET /` - API information and endpoint list
- `GET /health` - Health check with model status
- `GET /models` - List available models
- `GET /feature-columns` - Get required feature columns

### Prediction Endpoints

#### Single Prediction
```bash
POST /predict
Content-Type: application/json

{
  "features": {
    "feature1": 0.5,
    "feature2": 1.2,
    ...
  },
  "model_name": "RF",
  "threshold": 0.5  // optional
}
```

#### Batch Prediction
```bash
POST /predict/batch?model_name=RF&threshold=0.5
Content-Type: multipart/form-data

file: <your-data.csv>
```

Response includes:
- Predictions array (0 or 1)
- Probabilities array (0.0 to 1.0)
- Evaluation metrics (if CSV includes `incendio` column)
- Confusion matrix (if ground truth available)

#### Threshold Optimization
```bash
POST /optimize-threshold
Content-Type: multipart/form-data

file: <labeled-data.csv>  // must include 'incendio' column
```

Returns optimal threshold that maximizes F1 score with detailed metrics.

#### Model Comparison
```bash
POST /compare?threshold=0.5  // optional
Content-Type: multipart/form-data

file: <your-data.csv>
```

Compares predictions from **all three models** (RF, MLP, XGBoost) on the same data:
- Returns predictions and probabilities from each model
- Computes evaluation metrics if `incendio` column present
- Identifies best-performing model by F1 score
- Useful for model selection and performance comparison

## Data Format

### CSV Upload Requirements

- CSV format with header row
- Must include all required feature columns (see `/feature-columns` endpoint)
- Optional: `incendio` column (0 or 1) for evaluation metrics
- Automatically drops: geographic metadata, training-only columns

### Feature Columns

The API automatically extracts feature requirements from the reference dataset on startup. Use the `/feature-columns` endpoint to see the exact list.

## Models

All models are loaded from Cloudflare R2:

- **RF.pkl** - Random Forest Regressor
- **MLP.pkl** - Multi-Layer Perceptron (with StandardScaler)
- **XGBoost.pkl** - XGBoost Regressor

Models are cached in memory with 1-hour TTL for optimal performance.

## Configuration

Edit `config.py` to customize:

- `R2_BASE_URL` - Base URL for model storage
- `MODEL_METADATA` - Default thresholds and descriptions
- `MAX_FILE_SIZE_MB` - Maximum upload size (default: 100MB)
- `MODEL_CACHE_TTL_SECONDS` - Cache duration (default: 3600s)

## Architecture

```
api/
├── main.py              # FastAPI application & endpoints
├── models.py            # Pydantic request/response schemas
├── model_manager.py     # Model loading & caching
├── preprocessing.py     # Data preprocessing & validation
├── config.py            # Configuration settings
├── requirements.txt     # Python dependencies
└── railway.json         # Deployment configuration
```

## Error Handling

- **400 Bad Request**: Invalid input data, preprocessing errors
- **500 Internal Server Error**: Model loading failures, prediction errors

All errors return JSON with `error` and optional `detail` fields.

## Notes

- Models are preloaded on startup (may take 30-60 seconds)
- Reference dataset is downloaded once to extract feature schema
- File uploads are limited to 100MB by default
- CORS is enabled for all origins (configure for production)

## Testing

```bash
# Health check
curl http://localhost:8000/health

# List models
curl http://localhost:8000/models

# Get feature columns
curl http://localhost:8000/feature-columns

# Single prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": {...}, "model_name": "RF"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch?model_name=RF \
  -F "file=@data.csv"

# Compare all models
curl -X POST http://localhost:8000/compare \
  -F "file=@data.csv"
```

## Testing Scripts

Run included test scripts to verify functionality:

```bash
# Basic endpoint tests
python test_api.py

# Model comparison tests (requires numpy, pandas)
python test_comparison.py
```

## Documentation

Interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`
