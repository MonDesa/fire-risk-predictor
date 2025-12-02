"""
Fire Risk Predictor API
FastAPI backend for ML-based fire risk prediction
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import pandas as pd
import numpy as np
from typing import Optional, List
import io
import sys

from models import (
    PredictionRequest,
    SinglePredictionResponse,
    BatchPredictionResponse,
    ModelInfo,
    ModelsListResponse,
    HealthResponse,
    ThresholdOptimizationRequest,
    ThresholdOptimizationResponse,
    ModelComparisonResult,
    ComparisonResponse,
    SampleDataResponse,
)
from model_manager import model_manager
from preprocessing import (
    preprocess_for_inference,
    validate_feature_dict,
    prepare_features_from_csv,
    compute_metrics,
    PreprocessingError,
)
from config import MODEL_METADATA, MODEL_URLS, MAX_FILE_SIZE_MB
from sklearn.metrics import f1_score


def log(message: str):
    """Print with immediate flush for container logs"""
    print(message, flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup: preload models
    log("Starting up Fire Risk Predictor API...")
    log(f"Model URLs configured: {MODEL_URLS}")
    
    errors = await model_manager.preload_all_models()

    # Load reference dataset to get feature columns AND cache it
    try:
        import httpx
        from config import DATASET_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_REGION
        from model_manager import create_s3_auth_headers

        log(f"Loading reference dataset from: {DATASET_URL}")
        
        auth_headers = create_s3_auth_headers(
            DATASET_URL, MINIO_ACCESS_KEY, MINIO_SECRET_KEY, MINIO_REGION
        )
        
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.get(DATASET_URL, headers=auth_headers)
            response.raise_for_status()

            # Load full dataset and cache it
            df_full = pd.read_csv(io.BytesIO(response.content))
            df_full = df_full.dropna()  # Clean data once
            
            log(f"Dataset loaded: {len(df_full)} records")
            
            # Cache the cleaned dataset
            model_manager.set_dataset(df_full)
            
            # Extract feature columns
            features_df, _ = preprocess_for_inference(df_full.head(100))
            model_manager.set_feature_columns(list(features_df.columns))
            log(f"Feature columns extracted: {len(features_df.columns)} columns")

    except Exception as e:
        log(f"Warning: Could not load reference dataset: {e}")
        log("Feature validation will be limited without reference columns")

    if not errors:
        log("All models loaded successfully!")
    else:
        log(f"Some models failed to load: {errors}")

    yield

    # Shutdown
    log("Shutting down...")
    await model_manager.close()
app = FastAPI(
    title="Fire Risk Predictor API",
    description="ML-based fire risk prediction using Random Forest, MLP, and XGBoost models",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Fire Risk Predictor API",
        "version": "1.0.0",
        "description": "ML-based fire risk prediction service",
        "endpoints": {
            "health": "/health",
            "models": "/models",
            "predict_single": "/predict",
            "predict_batch": "/predict/batch",
            "optimize_threshold": "/optimize-threshold",
        },
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    models_status = {}

    for model_name in MODEL_URLS.keys():
        try:
            model = model_manager.cache.get(model_name)
            models_status[model_name] = model is not None
        except:
            models_status[model_name] = False

    all_loaded = all(models_status.values())

    return {
        "status": "healthy" if all_loaded else "degraded",
        "models_loaded": models_status,
    }
@app.get("/models", response_model=ModelsListResponse, tags=["Models"])
async def list_models():
    """List available models and their status"""
    models = []
    
    for model_key, metadata in MODEL_METADATA.items():
        try:
            # Check if model is cached
            cached = model_manager.cache.get(model_key) is not None
            loading = model_manager.cache.loading.get(model_key, False)
            
            if cached:
                status = "loaded"
            elif loading:
                status = "loading"
            else:
                status = "not_loaded"
        except:
            status = "error"
        
        models.append(
            ModelInfo(
                name=metadata["name"],
                model_key=model_key,
                description=metadata["description"],
                default_threshold=metadata["default_threshold"],
                status=status,
            )
        )
    
    return ModelsListResponse(models=models)


@app.post("/predict", response_model=SinglePredictionResponse, tags=["Prediction"])
async def predict_single(request: PredictionRequest):
    """
    Predict fire risk for a single data point
    
    - **features**: Dictionary of feature values
    - **model_name**: Model to use (RF, MLP, or XGBoost)
    - **threshold**: Optional custom threshold (default: model's default)
    """
    try:
        # Get model
        model = await model_manager.get_model(request.model_name)
        
        # Get reference columns
        reference_columns = model_manager.get_feature_columns()
        if reference_columns is None:
            raise HTTPException(
                status_code=500,
                detail="Reference feature columns not available. Server initialization incomplete."
            )
        
        # Validate and convert features to DataFrame
        features_df = validate_feature_dict(request.features, reference_columns)
        
        # Predict
        probability = float(model.predict(features_df)[0])
        
        # Apply threshold
        threshold = request.threshold if request.threshold is not None else MODEL_METADATA[request.model_name]["default_threshold"]
        prediction = int(probability >= threshold)
        
        return SinglePredictionResponse(
            model_name=request.model_name,
            threshold_used=threshold,
            prediction=prediction,
            probability=probability,
        )
        
    except PreprocessingError as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Prediction"])
async def predict_batch(
    file: UploadFile = File(...),
    model_name: str = Query(default="RF", regex="^(RF|MLP|XGBoost)$"),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
):
    """
    Predict fire risk for batch data from CSV file
    
    - **file**: CSV file with feature data
    - **model_name**: Model to use (RF, MLP, or XGBoost)
    - **threshold**: Optional custom threshold (default: model's default)
    
    If CSV includes 'incendio' column, evaluation metrics will be computed.
    """
    try:
        # Check file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read file
        content = await file.read()
        
        # Check file size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({size_mb:.2f}MB). Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
        
        # Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Get model
        model = await model_manager.get_model(model_name)
        
        # Get reference columns
        reference_columns = model_manager.get_feature_columns()
        if reference_columns is None:
            raise HTTPException(
                status_code=500,
                detail="Reference feature columns not available. Server initialization incomplete."
            )
        
        # Prepare features
        features_df, target = prepare_features_from_csv(df, reference_columns)
        
        if features_df.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid data remaining after preprocessing"
            )
        
        # Predict probabilities
        probabilities = model.predict(features_df)
        
        # Apply threshold
        threshold_used = threshold if threshold is not None else MODEL_METADATA[model_name]["default_threshold"]
        predictions = (probabilities >= threshold_used).astype(int)
        
        # Compute evaluation metrics if target is available
        evaluation_metrics = None
        confusion_matrix = None
        
        if target is not None:
            metrics = compute_metrics(target.values, predictions)
            evaluation_metrics = {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1_score": metrics["f1_score"],
            }
            confusion_matrix = metrics["confusion_matrix"]
        
        # Count predictions
        fire_count = int(np.sum(predictions))
        no_fire_count = len(predictions) - fire_count
        
        return BatchPredictionResponse(
            model_name=model_name,
            threshold_used=threshold_used,
            predictions=predictions.tolist(),
            probabilities=probabilities.tolist(),
            total_records=len(predictions),
            fire_predicted=fire_count,
            no_fire_predicted=no_fire_count,
            evaluation_metrics=evaluation_metrics,
            confusion_matrix=confusion_matrix,
        )
        
    except PreprocessingError as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.post("/optimize-threshold", response_model=ThresholdOptimizationResponse, tags=["Optimization"])
async def optimize_threshold(
    file: UploadFile = File(...),
    request: ThresholdOptimizationRequest = None,
):
    """
    Optimize classification threshold for maximum F1 score
    
    - **file**: CSV file with feature data and 'incendio' column (required for optimization)
    - **model_name**: Model to use
    - **threshold_min**: Minimum threshold to test (default: 0.1)
    - **threshold_max**: Maximum threshold to test (default: 0.9)
    - **threshold_step**: Step size for grid search (default: 0.05)
    
    Returns optimal threshold and corresponding metrics.
    """
    # Use defaults if request not provided
    if request is None:
        request = ThresholdOptimizationRequest()
    
    try:
        # Check file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read file
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Check for target column
        from config import TARGET_COLUMN
        if TARGET_COLUMN not in df.columns:
            raise HTTPException(
                status_code=400,
                detail=f"CSV must include '{TARGET_COLUMN}' column for threshold optimization"
            )
        
        # Get model
        model = await model_manager.get_model(request.model_name)
        
        # Get reference columns
        reference_columns = model_manager.get_feature_columns()
        if reference_columns is None:
            raise HTTPException(
                status_code=500,
                detail="Reference feature columns not available"
            )
        
        # Prepare features
        features_df, target = prepare_features_from_csv(df, reference_columns)
        
        if target is None:
            raise HTTPException(
                status_code=400,
                detail=f"Target column '{TARGET_COLUMN}' not found after preprocessing"
            )
        
        # Predict probabilities
        probabilities = model.predict(features_df)
        
        # Grid search for best threshold
        thresholds = np.arange(
            request.threshold_min,
            request.threshold_max + request.threshold_step,
            request.threshold_step
        )
        
        best_f1 = 0
        best_threshold = 0.5
        best_metrics = {}
        f1_scores = []
        
        for threshold in thresholds:
            predictions = (probabilities >= threshold).astype(int)
            metrics = compute_metrics(target.values, predictions)
            f1 = metrics["f1_score"]
            f1_scores.append(f1)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = float(threshold)
                best_metrics = metrics
        
        return ThresholdOptimizationResponse(
            model_name=request.model_name,
            optimal_threshold=best_threshold,
            f1_score=best_metrics["f1_score"],
            accuracy=best_metrics["accuracy"],
            precision=best_metrics["precision"],
            recall=best_metrics["recall"],
            thresholds_tested=thresholds.tolist(),
            f1_scores=f1_scores,
        )
        
    except PreprocessingError as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Threshold optimization failed: {str(e)}")


@app.get("/feature-columns", tags=["Info"])
async def get_feature_columns():
    """Get list of required feature columns"""
    columns = model_manager.get_feature_columns()
    
    if columns is None:
        raise HTTPException(
            status_code=500,
            detail="Feature columns not available. Server initialization incomplete."
        )
    
    return {
        "feature_columns": columns,
        "total_features": len(columns),
    }


@app.get("/sample", response_model=SampleDataResponse, tags=["Data"])
async def get_sample_data(
    size: int = Query(default=10000, ge=100, le=50000, description="Sample size (100-50000)"),
):
    """
    Get a random sample from the reference dataset for testing
    
    - **size**: Number of random rows to sample (default: 10000, max: 50000)
    
    Returns a balanced sample with both fire and non-fire records.
    Each call returns a different random sample.
    """
    try:
        from config import TARGET_COLUMN
        
        log(f"Generating random sample of {size} records...")
        
        # Use cached dataset
        df = model_manager.get_dataset()
        
        if df is None:
            raise HTTPException(
                status_code=503,
                detail="Dataset not loaded yet. Please wait for server initialization."
            )
        
        total_records = len(df)
        
        if total_records == 0:
            raise HTTPException(status_code=500, detail="Dataset is empty")
        
        # Create balanced sample if possible
        has_target = TARGET_COLUMN in df.columns
        
        if has_target:
            # Get balanced sample (50% fire, 50% non-fire)
            half_size = size // 2
            
            df_fire = df[df[TARGET_COLUMN] == 1]
            df_no_fire = df[df[TARGET_COLUMN] == 0]
            
            # Sample from each class
            fire_sample_size = min(half_size, len(df_fire))
            no_fire_sample_size = min(half_size, len(df_no_fire))
            
            df_fire_sample = df_fire.sample(n=fire_sample_size, random_state=None)
            df_no_fire_sample = df_no_fire.sample(n=no_fire_sample_size, random_state=None)
            
            # Combine and shuffle
            df_sample = pd.concat([df_fire_sample, df_no_fire_sample]).sample(frac=1)
        else:
            # Simple random sample
            sample_size = min(size, total_records)
            df_sample = df.sample(n=sample_size, random_state=None)
        
        # Get preview (first 10 rows) - this is what we show in the UI
        preview_df = df_sample.head(10)
        preview = preview_df.to_dict(orient='records')
        
        # Don't return full data - just keep sample indices for later use
        # Store the sample temporarily for the compare/sample endpoint
        sample_indices = df_sample.index.tolist()
        
        log(f"Sample generated: {len(df_sample)} records")
        
        return SampleDataResponse(
            total_records=total_records,
            sample_size=len(df_sample),
            columns=list(df_sample.columns),
            preview=preview,
            data=[],  # Don't return full data to save memory/bandwidth
            has_ground_truth=has_target,
        )
            
    except HTTPException:
        raise
    except Exception as e:
        log(f"Error getting sample: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sample: {str(e)}")


@app.post("/compare/sample", response_model=ComparisonResponse, tags=["Prediction"])
async def compare_models_with_sample(
    size: int = Query(default=10000, ge=100, le=50000, description="Sample size"),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
):
    """
    Compare all models using a random sample from the reference dataset
    
    - **size**: Sample size (default: 10000)
    - **threshold**: Optional custom threshold for all models
    
    Uses cached dataset and runs predictions with all models.
    Since the dataset includes ground truth, evaluation metrics are always computed.
    """
    try:
        from config import TARGET_COLUMN
        
        log(f"Running comparison on random sample of {size} records...")
        
        # Use cached dataset
        df = model_manager.get_dataset()
        
        if df is None:
            raise HTTPException(
                status_code=503,
                detail="Dataset not loaded yet. Please wait for server initialization."
            )
        
        # Create balanced sample
        half_size = size // 2
        df_fire = df[df[TARGET_COLUMN] == 1]
        df_no_fire = df[df[TARGET_COLUMN] == 0]
        
        fire_sample_size = min(half_size, len(df_fire))
        no_fire_sample_size = min(half_size, len(df_no_fire))
        
        df_fire_sample = df_fire.sample(n=fire_sample_size, random_state=None)
        df_no_fire_sample = df_no_fire.sample(n=no_fire_sample_size, random_state=None)
        
        df_sample = pd.concat([df_fire_sample, df_no_fire_sample]).sample(frac=1)
        
        # Get reference columns
        reference_columns = model_manager.get_feature_columns()
        if reference_columns is None:
            raise HTTPException(
                status_code=500,
                detail="Reference feature columns not available"
            )
        
        # Prepare features
        features_df, target = prepare_features_from_csv(df_sample, reference_columns)
        
        if features_df.empty:
            raise HTTPException(status_code=500, detail="No valid data after preprocessing")
        
        total_records = len(features_df)
        has_ground_truth = target is not None
        results = []
        best_model_name = None
        best_f1 = -1.0
        
        # Run predictions for each model
        for model_name in ["RF", "MLP", "XGBoost"]:
            try:
                model = await model_manager.get_model(model_name)
                probabilities = model.predict(features_df)
                
                threshold_used = threshold if threshold is not None else MODEL_METADATA[model_name]["default_threshold"]
                predictions = (probabilities >= threshold_used).astype(int)
                
                fire_count = int(np.sum(predictions))
                no_fire_count = len(predictions) - fire_count
                
                evaluation_metrics = None
                confusion_matrix_result = None
                
                if has_ground_truth:
                    metrics = compute_metrics(target.values, predictions)
                    evaluation_metrics = {
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                    }
                    confusion_matrix_result = metrics["confusion_matrix"]
                    
                    if metrics["f1_score"] > best_f1:
                        best_f1 = metrics["f1_score"]
                        best_model_name = model_name
                
                results.append(ModelComparisonResult(
                    model_name=model_name,
                    threshold_used=threshold_used,
                    predictions=predictions.tolist(),
                    probabilities=probabilities.tolist(),
                    fire_predicted=fire_count,
                    no_fire_predicted=no_fire_count,
                    evaluation_metrics=evaluation_metrics,
                    confusion_matrix=confusion_matrix_result,
                ))
                
            except Exception as e:
                results.append(ModelComparisonResult(
                    model_name=model_name,
                    threshold_used=threshold if threshold is not None else MODEL_METADATA[model_name]["default_threshold"],
                    predictions=[],
                    probabilities=[],
                    fire_predicted=0,
                    no_fire_predicted=0,
                    evaluation_metrics={"error": str(e)},
                    confusion_matrix=None,
                ))
        
        log(f"Sample comparison completed. Best model: {best_model_name}")
        
        return ComparisonResponse(
            total_records=total_records,
            has_ground_truth=has_ground_truth,
            results=results,
            best_model=best_model_name,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        log(f"Error in sample comparison: {e}")
        raise HTTPException(status_code=500, detail=f"Sample comparison failed: {str(e)}")


@app.post("/compare", response_model=ComparisonResponse, tags=["Prediction"])
async def compare_models(
    file: UploadFile = File(...),
    threshold: Optional[float] = Query(default=None, ge=0.0, le=1.0),
):
    """
    Compare predictions from all three models (RF, MLP, XGBoost)
    
    - **file**: CSV file with feature data
    - **threshold**: Optional custom threshold for all models (default: each model's default)
    
    If CSV includes 'incendio' column, evaluation metrics will be computed and
    the best-performing model (by F1 score) will be identified.
    
    Returns predictions from all models for comparison.
    """
    try:
        # Check file type
        if not file.filename.endswith('.csv'):
            raise HTTPException(status_code=400, detail="File must be a CSV")
        
        # Read file
        content = await file.read()
        
        # Check file size
        size_mb = len(content) / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({size_mb:.2f}MB). Maximum size: {MAX_FILE_SIZE_MB}MB"
            )
        
        # Parse CSV
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {str(e)}")
        
        if df.empty:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        
        # Get reference columns
        reference_columns = model_manager.get_feature_columns()
        if reference_columns is None:
            raise HTTPException(
                status_code=500,
                detail="Reference feature columns not available. Server initialization incomplete."
            )
        
        # Prepare features (once for all models)
        features_df, target = prepare_features_from_csv(df, reference_columns)
        
        if features_df.empty:
            raise HTTPException(
                status_code=400,
                detail="No valid data remaining after preprocessing"
            )
        
        total_records = len(features_df)
        has_ground_truth = target is not None
        results = []
        best_model_name = None
        best_f1 = -1.0
        
        # Run predictions for each model
        for model_name in ["RF", "MLP", "XGBoost"]:
            try:
                # Get model
                model = await model_manager.get_model(model_name)
                
                # Predict probabilities
                probabilities = model.predict(features_df)
                
                # Apply threshold
                threshold_used = threshold if threshold is not None else MODEL_METADATA[model_name]["default_threshold"]
                predictions = (probabilities >= threshold_used).astype(int)
                
                # Count predictions
                fire_count = int(np.sum(predictions))
                no_fire_count = len(predictions) - fire_count
                
                # Compute evaluation metrics if target is available
                evaluation_metrics = None
                confusion_matrix_result = None
                
                if has_ground_truth:
                    metrics = compute_metrics(target.values, predictions)
                    evaluation_metrics = {
                        "accuracy": metrics["accuracy"],
                        "precision": metrics["precision"],
                        "recall": metrics["recall"],
                        "f1_score": metrics["f1_score"],
                    }
                    confusion_matrix_result = metrics["confusion_matrix"]
                    
                    # Track best model by F1 score
                    if metrics["f1_score"] > best_f1:
                        best_f1 = metrics["f1_score"]
                        best_model_name = model_name
                
                # Add to results
                results.append(ModelComparisonResult(
                    model_name=model_name,
                    threshold_used=threshold_used,
                    predictions=predictions.tolist(),
                    probabilities=probabilities.tolist(),
                    fire_predicted=fire_count,
                    no_fire_predicted=no_fire_count,
                    evaluation_metrics=evaluation_metrics,
                    confusion_matrix=confusion_matrix_result,
                ))
                
            except Exception as e:
                # If one model fails, add error result
                results.append(ModelComparisonResult(
                    model_name=model_name,
                    threshold_used=threshold if threshold is not None else MODEL_METADATA[model_name]["default_threshold"],
                    predictions=[],
                    probabilities=[],
                    fire_predicted=0,
                    no_fire_predicted=0,
                    evaluation_metrics={"error": str(e)} if has_ground_truth else None,
                    confusion_matrix=None,
                ))
        
        return ComparisonResponse(
            total_records=total_records,
            has_ground_truth=has_ground_truth,
            results=results,
            best_model=best_model_name,
        )
        
    except PreprocessingError as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison failed: {str(e)}")