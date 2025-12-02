"""
Model management - loading and caching models from R2
"""
import joblib
import httpx
import io
from typing import Dict, Optional, Any
from datetime import datetime, timedelta
import asyncio
from config import MODEL_URLS, MODEL_METADATA, MODEL_CACHE_TTL_SECONDS


class ModelCache:
    """Simple in-memory model cache with TTL"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.load_times: Dict[str, datetime] = {}
        self.feature_columns: Optional[list] = None
        self._lock = asyncio.Lock()
    
    def is_expired(self, model_name: str) -> bool:
        """Check if cached model is expired"""
        if model_name not in self.load_times:
            return True
        
        age = datetime.now() - self.load_times[model_name]
        return age.total_seconds() > MODEL_CACHE_TTL_SECONDS
    
    def get(self, model_name: str) -> Optional[Any]:
        """Get model from cache if valid"""
        if model_name not in self.models or self.is_expired(model_name):
            return None
        return self.models[model_name]
    
    def set(self, model_name: str, model: Any):
        """Store model in cache"""
        self.models[model_name] = model
        self.load_times[model_name] = datetime.now()
    
    def clear(self, model_name: Optional[str] = None):
        """Clear cache for specific model or all models"""
        if model_name:
            self.models.pop(model_name, None)
            self.load_times.pop(model_name, None)
        else:
            self.models.clear()
            self.load_times.clear()


class ModelManager:
    """Manages loading and caching of ML models"""
    
    def __init__(self):
        self.cache = ModelCache()
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def load_model_from_url(self, url: str) -> Any:
        """
        Load model from R2 URL
        
        Args:
            url: Model URL
            
        Returns:
            Loaded model object
            
        Raises:
            Exception: If loading fails
        """
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            # Load model from bytes
            model_bytes = io.BytesIO(response.content)
            model = joblib.load(model_bytes)
            
            return model
            
        except httpx.HTTPError as e:
            raise Exception(f"Failed to download model from {url}: {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    async def get_model(self, model_name: str) -> Any:
        """
        Get model, loading from R2 if not cached
        
        Args:
            model_name: Model key (RF, MLP, or XGBoost)
            
        Returns:
            Loaded model object
            
        Raises:
            ValueError: If model_name is invalid
            Exception: If loading fails
        """
        if model_name not in MODEL_URLS:
            raise ValueError(
                f"Invalid model name: {model_name}. "
                f"Available models: {list(MODEL_URLS.keys())}"
            )
        
        # Check cache first
        async with self.cache._lock:
            cached_model = self.cache.get(model_name)
            if cached_model is not None:
                return cached_model
            
            # Load from R2
            print(f"Loading model {model_name} from R2...")
            url = MODEL_URLS[model_name]
            model = await self.load_model_from_url(url)
            
            # Cache it
            self.cache.set(model_name, model)
            print(f"Model {model_name} loaded and cached successfully")
            
            return model
    
    async def preload_all_models(self):
        """Preload all models into cache"""
        print("Preloading all models...")
        errors = {}
        
        for model_name in MODEL_URLS.keys():
            try:
                await self.get_model(model_name)
            except Exception as e:
                errors[model_name] = str(e)
                print(f"Failed to preload {model_name}: {str(e)}")
        
        if errors:
            print(f"Preload completed with errors: {errors}")
        else:
            print("All models preloaded successfully")
        
        return errors
    
    def get_model_info(self, model_name: str) -> dict:
        """Get metadata about a model"""
        if model_name not in MODEL_METADATA:
            raise ValueError(f"Unknown model: {model_name}")
        
        info = MODEL_METADATA[model_name].copy()
        info["model_key"] = model_name
        info["cached"] = model_name in self.cache.models
        info["cache_expired"] = self.cache.is_expired(model_name)
        
        return info
    
    async def close(self):
        """Clean up resources"""
        await self.client.aclose()
    
    def set_feature_columns(self, columns: list):
        """Store reference feature columns"""
        self.cache.feature_columns = columns
    
    def get_feature_columns(self) -> Optional[list]:
        """Get reference feature columns"""
        return self.cache.feature_columns


# Global model manager instance
model_manager = ModelManager()
