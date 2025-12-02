"""
Data preprocessing utilities for Fire Risk Predictor
"""
import pandas as pd
import numpy as np
from typing import Tuple, Optional, List
from config import (
    COLUMNS_TO_DROP_INFERENCE,
    OPTIONAL_COLUMNS_TO_DROP,
    TARGET_COLUMN,
)


class PreprocessingError(Exception):
    """Custom exception for preprocessing errors"""
    pass


def validate_dataframe(df: pd.DataFrame, require_target: bool = False) -> None:
    """
    Validate input dataframe
    
    Args:
        df: Input dataframe
        require_target: Whether to require the target column
        
    Raises:
        PreprocessingError: If validation fails
    """
    if df.empty:
        raise PreprocessingError("DataFrame is empty")
    
    if require_target and TARGET_COLUMN not in df.columns:
        raise PreprocessingError(f"Target column '{TARGET_COLUMN}' not found in data")
    
    # Check for required columns after dropping
    drop_cols = [col for col in COLUMNS_TO_DROP_INFERENCE if col in df.columns]
    optional_drops = [col for col in OPTIONAL_COLUMNS_TO_DROP if col in df.columns]
    
    # Add target column to drops if present and not required
    if not require_target and TARGET_COLUMN in df.columns:
        drop_cols.append(TARGET_COLUMN)
    
    remaining_cols = set(df.columns) - set(drop_cols) - set(optional_drops)
    if require_target:
        remaining_cols.discard(TARGET_COLUMN)
    
    if len(remaining_cols) == 0:
        raise PreprocessingError(
            f"No feature columns remaining after dropping: {drop_cols + optional_drops}"
        )


def preprocess_for_inference(
    df: pd.DataFrame,
    drop_nulls: bool = True,
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Preprocess dataframe for inference
    
    Args:
        df: Input dataframe
        drop_nulls: Whether to drop rows with null values
        
    Returns:
        Tuple of (features_df, target_series or None)
        
    Raises:
        PreprocessingError: If preprocessing fails
    """
    df = df.copy()
    
    # Check if target column exists (for evaluation)
    has_target = TARGET_COLUMN in df.columns
    target = df[TARGET_COLUMN].copy() if has_target else None
    
    # Validate before processing
    validate_dataframe(df, require_target=False)
    
    # Drop columns that should not be used for modeling
    columns_to_drop = []
    
    # Always drop these if present
    for col in COLUMNS_TO_DROP_INFERENCE:
        if col in df.columns:
            columns_to_drop.append(col)
    
    # Drop optional columns
    for col in OPTIONAL_COLUMNS_TO_DROP:
        if col in df.columns:
            columns_to_drop.append(col)
    
    # Drop target column for features
    if TARGET_COLUMN in df.columns:
        columns_to_drop.append(TARGET_COLUMN)
    
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Handle null values
    if drop_nulls:
        null_count = df.isnull().sum().sum()
        if null_count > 0:
            original_len = len(df)
            df = df.dropna()
            if target is not None:
                target = target.loc[df.index]
            dropped = original_len - len(df)
            if dropped > 0:
                print(f"Dropped {dropped} rows with null values")
    else:
        # If not dropping nulls, raise error if any exist
        if df.isnull().sum().sum() > 0:
            raise PreprocessingError(
                "Data contains null values. Please handle them or set drop_nulls=True"
            )
    
    if df.empty:
        raise PreprocessingError("No data remaining after preprocessing")
    
    return df, target


def validate_feature_dict(features: dict, reference_columns: List[str]) -> pd.DataFrame:
    """
    Validate and convert a feature dictionary to DataFrame
    
    Args:
        features: Dictionary of feature values
        reference_columns: Expected column names
        
    Returns:
        DataFrame with single row
        
    Raises:
        PreprocessingError: If validation fails
    """
    if not features:
        raise PreprocessingError("Feature dictionary is empty")
    
    # Create DataFrame from dict
    try:
        df = pd.DataFrame([features])
    except Exception as e:
        raise PreprocessingError(f"Failed to create DataFrame from features: {str(e)}")
    
    # Check for missing columns
    missing_cols = set(reference_columns) - set(df.columns)
    if missing_cols:
        raise PreprocessingError(
            f"Missing required columns: {sorted(missing_cols)}"
        )
    
    # Check for extra columns
    extra_cols = set(df.columns) - set(reference_columns)
    if extra_cols:
        raise PreprocessingError(
            f"Unexpected columns found: {sorted(extra_cols)}"
        )
    
    # Ensure correct column order
    df = df[reference_columns]
    
    return df


def prepare_features_from_csv(
    df: pd.DataFrame,
    reference_columns: List[str],
) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Prepare features from uploaded CSV
    
    Args:
        df: Input dataframe
        reference_columns: Expected feature columns after preprocessing
        
    Returns:
        Tuple of (features_df, target_series or None)
    """
    # Preprocess
    features_df, target = preprocess_for_inference(df, drop_nulls=True)
    
    # Check if columns match expected
    missing_cols = set(reference_columns) - set(features_df.columns)
    if missing_cols:
        raise PreprocessingError(
            f"CSV is missing required feature columns: {sorted(missing_cols)}"
        )
    
    # Remove extra columns not in reference
    extra_cols = set(features_df.columns) - set(reference_columns)
    if extra_cols:
        features_df = features_df.drop(columns=list(extra_cols))
    
    # Ensure correct column order
    features_df = features_df[reference_columns]
    
    return features_df, target


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict:
    """
    Compute evaluation metrics
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary of metrics
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
    )
    
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }
