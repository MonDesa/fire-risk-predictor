"""
Pydantic models for request/response validation
"""
from typing import List, Optional, Dict, Literal
from pydantic import BaseModel, Field, field_validator, ConfigDict


class PredictionRequest(BaseModel):
    """Single row prediction request"""
    model_config = ConfigDict(protected_namespaces=())
    
    features: Dict[str, float] = Field(
        ..., 
        description="Feature dictionary with column names as keys"
    )
    model_name: Literal["RF", "MLP", "XGBoost"] = Field(
        default="RF",
        description="Model to use for prediction"
    )
    threshold: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Custom threshold for binary classification. If not provided, uses model default."
    )

    @field_validator('threshold')
    @classmethod
    def validate_threshold(cls, v):
        if v is not None and (v < 0.0 or v > 1.0):
            raise ValueError('Threshold must be between 0.0 and 1.0')
        return v


class BatchPredictionResponse(BaseModel):
    """Batch prediction response"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    threshold_used: float
    predictions: List[int] = Field(description="Binary predictions (0 or 1)")
    probabilities: List[float] = Field(description="Fire risk probabilities (0.0 to 1.0)")
    total_records: int
    fire_predicted: int
    no_fire_predicted: int
    evaluation_metrics: Optional[Dict[str, float]] = Field(
        default=None,
        description="Metrics if ground truth column was provided"
    )
    confusion_matrix: Optional[List[List[int]]] = Field(
        default=None,
        description="Confusion matrix if ground truth was provided [[TN, FP], [FN, TP]]"
    )


class SinglePredictionResponse(BaseModel):
    """Single prediction response"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    threshold_used: float
    prediction: int = Field(description="Binary prediction (0 or 1)")
    probability: float = Field(description="Fire risk probability (0.0 to 1.0)")


class ModelInfo(BaseModel):
    """Model information"""
    model_config = ConfigDict(protected_namespaces=())
    
    name: str
    model_key: str
    description: str
    default_threshold: float
    status: str = Field(description="loaded or error")


class ModelsListResponse(BaseModel):
    """List of available models"""
    models: List[ModelInfo]


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    models_loaded: Dict[str, bool]


class ErrorResponse(BaseModel):
    """Error response"""
    error: str
    detail: Optional[str] = None


class ThresholdOptimizationRequest(BaseModel):
    """Request for threshold optimization"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: Literal["RF", "MLP", "XGBoost"] = Field(
        default="RF",
        description="Model to use for optimization"
    )
    threshold_min: float = Field(default=0.1, ge=0.0, le=1.0)
    threshold_max: float = Field(default=0.9, ge=0.0, le=1.0)
    threshold_step: float = Field(default=0.05, ge=0.01, le=0.1)

    @field_validator('threshold_max')
    @classmethod
    def validate_threshold_range(cls, v, info):
        if 'threshold_min' in info.data and v <= info.data['threshold_min']:
            raise ValueError('threshold_max must be greater than threshold_min')
        return v


class ThresholdOptimizationResponse(BaseModel):
    """Response from threshold optimization"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    optimal_threshold: float
    f1_score: float
    accuracy: float
    precision: float
    recall: float
    thresholds_tested: List[float]
    f1_scores: List[float]


class ModelComparisonResult(BaseModel):
    """Results for a single model in comparison"""
    model_config = ConfigDict(protected_namespaces=())
    
    model_name: str
    threshold_used: float
    predictions: List[int]
    probabilities: List[float]
    fire_predicted: int
    no_fire_predicted: int
    evaluation_metrics: Optional[Dict[str, float]] = None
    confusion_matrix: Optional[List[List[int]]] = None


class ComparisonResponse(BaseModel):
    """Response comparing all models"""
    total_records: int
    has_ground_truth: bool
    results: List[ModelComparisonResult]
    best_model: Optional[str] = Field(
        default=None,
        description="Model with highest F1 score (only if ground truth provided)"
    )


class SampleDataResponse(BaseModel):
    """Response with sample data from the reference dataset"""
    total_records: int
    sample_size: int
    columns: List[str]
    preview: List[Dict] = Field(
        description="First 10 rows for preview display"
    )
    data: List[Dict] = Field(
        description="Full sample data as list of dicts"
    )
    has_ground_truth: bool = Field(
        description="Whether the sample includes 'incendio' column"
    )
