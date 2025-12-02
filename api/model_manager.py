"""Model management - loading and caching models from MinIO"""

import joblib
import httpx
import io
from typing import Dict, Optional, Any
from datetime import datetime
import asyncio
from config import MODEL_URLS, MODEL_METADATA, MODEL_CACHE_TTL_SECONDS


class ModelCache:
    """Simple in-memory model cache with TTL"""
    
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.load_times: Dict[str, datetime] = {}
        self.loading: Dict[str, bool] = {}  # Track which models are currently loading
        self.feature_columns: Optional[list] = None
        self._locks: Dict[str, asyncio.Lock] = {}  # Per-model locks
    
    def _get_lock(self, model_name: str) -> asyncio.Lock:
        """Get or create a lock for a specific model"""
        if model_name not in self._locks:
            self._locks[model_name] = asyncio.Lock()
        return self._locks[model_name]
    
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
        self.loading[model_name] = False
    
    def clear(self, model_name: Optional[str] = None):
        """Clear cache for specific model or all models"""
        if model_name:
            self.models.pop(model_name, None)
            self.load_times.pop(model_name, None)
            self.loading.pop(model_name, None)
        else:
            self.models.clear()
            self.load_times.clear()
            self.loading.clear()


class ModelManager:
    """Manages loading and caching of ML models from MinIO via HTTP"""
    
    def __init__(self):
        self.cache = ModelCache()
        self.http_client: Optional[httpx.AsyncClient] = None
    
    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client"""
        if self.http_client is None or self.http_client.is_closed:
            self.http_client = httpx.AsyncClient(timeout=300.0)  # 5 min timeout for large models
        return self.http_client
    
    async def load_model_from_url(self, url: str, model_name: str) -> Any:
        """
        Load model from MinIO via HTTP
        
        Args:
            url: Full URL to the model file
            model_name: Model name for logging
            
        Returns:
            Loaded model object
            
        Raises:
            Exception: If loading fails
        """
        try:
            print(f"[{model_name}] Downloading from {url}...")
            start_time = datetime.now()
            
            client = await self._get_client()
            response = await client.get(url)
            response.raise_for_status()
            
            content_length = len(response.content)
            download_time = (datetime.now() - start_time).total_seconds()
            print(f"[{model_name}] Downloaded {content_length / (1024*1024):.2f} MB in {download_time:.1f}s, loading into memory...")
            
            # Load model from bytes
            model_bytes = io.BytesIO(response.content)
            model = joblib.load(model_bytes)
            
            total_time = (datetime.now() - start_time).total_seconds()
            print(f"[{model_name}] Model loaded successfully in {total_time:.1f}s total!")
            return model
            
        except httpx.HTTPStatusError as e:
            raise Exception(f"Failed to download model (HTTP {e.response.status_code}): {str(e)}")
        except Exception as e:
            raise Exception(f"Failed to load model: {str(e)}")
    
    async def get_model(self, model_name: str) -> Any:
        """
        Get model, loading from MinIO if not cached
        
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
        
        # Check cache first (no lock needed for read)
        cached_model = self.cache.get(model_name)
        if cached_model is not None:
            return cached_model
        
        # Use per-model lock to prevent duplicate loading
        lock = self.cache._get_lock(model_name)
        
        async with lock:
            # Double-check cache after acquiring lock
            cached_model = self.cache.get(model_name)
            if cached_model is not None:
                return cached_model
            
            # Mark as loading
            self.cache.loading[model_name] = True
            
            try:
                # Load from MinIO via HTTP
                url = MODEL_URLS[model_name]
                model = await self.load_model_from_url(url, model_name)
                
                # Cache it
                self.cache.set(model_name, model)
                
                return model
            except Exception as e:
                self.cache.loading[model_name] = False
                raise
    
    async def preload_all_models(self):
        """Preload all models into cache concurrently"""
        print("Preloading all models concurrently from MinIO...")
        
        async def load_single(model_name: str) -> tuple[str, Optional[str]]:
            try:
                await self.get_model(model_name)
                return (model_name, None)
            except Exception as e:
                error_msg = str(e)
                print(f"Failed to preload {model_name}: {error_msg}")
                return (model_name, error_msg)
        
        # Load all models concurrently
        results = await asyncio.gather(*[
            load_single(name) for name in MODEL_URLS.keys()
        ])
        
        errors = {name: err for name, err in results if err is not None}
        
        if errors:
            print(f"Preload completed with errors: {errors}")
        else:
            print("All models preloaded successfully from MinIO")
        
        return errors
    
    def get_model_info(self, model_name: str) -> dict:
        """Get metadata about a model"""
        if model_name not in MODEL_METADATA:
            raise ValueError(f"Unknown model: {model_name}")
        
        info = MODEL_METADATA[model_name].copy()
        info["model_key"] = model_name
        info["cached"] = model_name in self.cache.models
        info["cache_expired"] = self.cache.is_expired(model_name)
        info["loading"] = self.cache.loading.get(model_name, False)
        
        return info
    
    async def close(self):
        """Clean up resources"""
        if self.http_client is not None and not self.http_client.is_closed:
            await self.http_client.aclose()
    
    def set_feature_columns(self, columns: list):
        """Store reference feature columns"""
        self.cache.feature_columns = columns
    
    def get_feature_columns(self) -> Optional[list]:
        """Get reference feature columns"""
        return self.cache.feature_columns


# Global model manager instance
model_manager = ModelManager()
