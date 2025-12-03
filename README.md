# Fire Risk Predictor

A machine learning-based system for predicting wildfire risk in Brazil using environmental and atmospheric data. The project consists of a trained ML pipeline, a FastAPI backend for model inference, and a React frontend for interactive predictions.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Machine Learning Models](#machine-learning-models)
- [API Reference](#api-reference)
- [Frontend](#frontend)
- [Installation](#installation)
- [Deployment](#deployment)
- [Environment Variables](#environment-variables)
- [License](#license)

## Overview

Brazil faces an increasing wildfire crisis, with 278,229 heat spots recorded in 2024 alone, the highest number in 14 years according to INPE. The Amazon region was the most affected biome, with over 140,000 spots representing a 77% increase compared to 2023.

This project addresses the challenge of reactive fire detection by using machine learning to predict where fires are most likely to occur based on environmental conditions, enabling preventive action before fires spread.

### Key Features

- Three trained ML models: Random Forest, MLP (Neural Network), and XGBoost
- REST API for batch predictions and model comparison
- Threshold optimization for maximizing F1 score
- Web interface for interactive predictions
- Support for both file upload and sample data testing

## Project Structure

```
fire-risk-predictor/
├── FireRiskPredictor.ipynb    # Training notebook with K-Fold validation
├── api/                        # FastAPI backend
│   ├── main.py                # API endpoints
│   ├── models.py              # Pydantic models
│   ├── model_manager.py       # Model loading and caching
│   ├── preprocessing.py       # Data preprocessing utilities
│   ├── config.py              # Configuration settings
│   ├── Dockerfile             # Container configuration
│   └── requirements.txt       # Python dependencies
├── frontend/                   # React frontend
│   ├── src/
│   │   ├── App.tsx            # Main application component
│   │   ├── components/        # UI components
│   │   └── services/          # API client
│   ├── package.json           # Node.js dependencies
│   └── vite.config.ts         # Vite configuration
└── README.md
```

## Dataset

The project uses data from SISAM (Sistema de Informacoes de Saude Ambiental), containing heat spot records across Brazil since 2003.

### Dataset Statistics

- Original dataset: 2.6+ million records
- Training sample: 250,000 balanced records (125k fire, 125k no-fire)
- Test sample: 50,000 records (separate from training)

### Features (14 variables)

| Feature | Description | Unit |
|---------|-------------|------|
| datahora | Date and time of observation | datetime |
| longitude | Geographic longitude | degrees |
| latitude | Geographic latitude | degrees |
| co_ppb | Carbon monoxide concentration | ppb |
| no2_ppb | Nitrogen dioxide concentration | ppb |
| o3_ppb | Ozone concentration | ppb |
| pm25_ugm3 | Particulate matter PM2.5 | ug/m3 |
| so2_ugm3 | Sulfur dioxide concentration | ug/m3 |
| precipitacao_mmdia | Daily precipitation | mm/day |
| temperatura_c | Temperature | Celsius |
| umidade_relativa_percentual | Relative humidity | % |
| vento_direcao_grau | Wind direction | degrees |
| vento_velocidade_ms | Wind speed | m/s |
| incendio | Fire occurrence (target) | 0 or 1 |

### Preprocessing

During inference, the following columns are excluded from model input:
- Location identifiers: datahora, longitude, latitude
- Categorical data: satelite, pais, estado, municipio, bioma
- Derived features: data_pas, numero_dias_sem_chuva, risco_fogo, frp, time_diff_hours

## Machine Learning Models

### Training Methodology

- K-Fold Cross Validation (K=3 and K=5)
- Balanced sampling to handle class imbalance
- StandardScaler preprocessing for MLP
- Grid search for optimal classification threshold

### Models

| Model | Description | Key Parameters |
|-------|-------------|----------------|
| Random Forest | Ensemble of decision trees | Default sklearn parameters |
| MLP | Neural network with 2 hidden layers | 100x100 neurons, max_iter=200 |
| XGBoost | Gradient boosting | n_estimators=100, reg:squarederror |

### Evaluation Metrics

Models are evaluated using:
- Accuracy
- Precision
- Recall
- F1 Score (primary metric for threshold optimization)
- Confusion Matrix

## API Reference

Base URL: `http://localhost:8000`

### Endpoints

#### Health Check
```
GET /health
```
Returns API status and model loading state.

#### List Models
```
GET /models
```
Returns available models with metadata.

#### Batch Prediction
```
POST /predict/batch
```
Upload a CSV file for batch predictions with a single model.

Parameters:
- `file`: CSV file (multipart/form-data)
- `model_name`: RF, MLP, or XGBoost (default: RF)
- `threshold`: Classification threshold 0.0-1.0 (optional)

#### Prediction with Sample Data
```
POST /predict/sample
```
Run predictions using cached sample dataset.

Parameters:
- `model_name`: RF, MLP, or XGBoost
- `size`: Sample size 100-50000 (default: 10000)
- `threshold`: Classification threshold (optional)

#### Compare Models
```
POST /compare
```
Compare all models on the same dataset.

Parameters:
- `file`: CSV file (multipart/form-data)
- `threshold`: Classification threshold (optional)

#### Compare Models with Sample
```
POST /compare/sample
```
Compare all models using cached sample dataset.

Parameters:
- `size`: Sample size 100-50000
- `threshold`: Classification threshold (optional)

#### Optimize Threshold
```
POST /optimize-threshold
```
Find optimal classification threshold for a model.

Parameters:
- `file`: CSV file with ground truth (multipart/form-data)
- `model_name`: RF, MLP, or XGBoost
- `threshold_min`: Minimum threshold (default: 0.1)
- `threshold_max`: Maximum threshold (default: 0.9)
- `threshold_step`: Step size (default: 0.05)

#### Optimize Threshold with Sample
```
POST /optimize-threshold/sample
```
Find optimal threshold using cached sample dataset.

Body (JSON):
```json
{
  "model_name": "RF",
  "sample_size": 10000,
  "threshold_min": 0.1,
  "threshold_max": 0.9,
  "threshold_step": 0.05
}
```

#### Get Sample Data
```
GET /sample
```
Retrieve a random sample from the reference dataset.

Parameters:
- `size`: Sample size 100-50000 (default: 10000)

#### Feature Columns
```
GET /feature-columns
```
Returns the list of required feature columns for prediction.

## Frontend

The web interface is built with React 19, TypeScript, and Tailwind CSS v4.

### Features

- Model status monitoring
- Data source toggle (file upload or sample data)
- Single model prediction
- Multi-model comparison
- Threshold optimization with visualization
- Confusion matrix display
- F1 score vs threshold charts

### Running Locally

```bash
cd frontend
npm install
npm run dev
```

The frontend will be available at `http://localhost:5173`.

## Installation

### Prerequisites

- Python 3.11+
- Node.js 18+
- Trained model files (RF.pkl, MLP.pkl, XGBoost.pkl)

### Backend Setup

```bash
cd api
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Frontend Setup

```bash
cd frontend
npm install
```

### Running Locally

Backend:
```bash
cd api
hypercorn main:app --bind 0.0.0.0:8000
```

Frontend:
```bash
cd frontend
npm run dev
```

## Deployment

The project is configured for deployment on Railway with the following services:

- API: Docker container with Hypercorn ASGI server
- Frontend: Static site deployment
- Storage: MinIO bucket for model files and sample dataset

### Docker Build (API)

```bash
cd api
docker build -t fire-risk-predictor-api .
docker run -p 8000:8000 --env-file .env fire-risk-predictor-api
```

## Environment Variables

### Backend (api/.env)

| Variable | Description | Default |
|----------|-------------|---------|
| MINIO_BASE_URL | MinIO server URL | http://bucket.railway.internal:9000 |
| MINIO_ACCESS_KEY | MinIO access key | (required) |
| MINIO_SECRET_KEY | MinIO secret key | (required) |
| MINIO_REGION | MinIO region | us-east-1 |

### Frontend (frontend/.env)

| Variable | Description | Default |
|----------|-------------|---------|
| VITE_API_URL | Backend API URL | http://localhost:8000 |

## Tech Stack

### Backend
- FastAPI 0.100.0
- Hypercorn 0.14.4
- Pandas 2.1+
- NumPy 2.0+
- Scikit-learn 1.3+
- XGBoost 2.0+
- Pydantic 2.3.0

### Frontend
- React 19.2
- TypeScript 5.9
- Vite 7.2
- Tailwind CSS 4.1
- Axios 1.13
- Lucide React (icons)

## Contributors

- [Gustavo Lelli](https://github.com/gustavo-lelli)
- [Otavio Coletti](https://github.com/otaviofcoletti)

## License

This project is developed by MonDesa.
