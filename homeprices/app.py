from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional,List
import os
from contextlib import asynccontextmanager
from model_manager import ModelManager
from data_enricher import Featurizer
from loguru import logger 



# Global instances
model_manager = ModelManager()
featurizer = Featurizer()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    # Load model on startup
    model_loaded = model_manager.load_model()
    if model_loaded:
        logger.info(f"Model loaded successfully: {model_manager.get_version()}")
    else:
        logger.error("No model loaded - predictions will fail until model is available")
    
    # Initialize data enricher
    await featurizer.initialize()
    
    yield
    
    # Cleanup
    await featurizer.cleanup()

app = FastAPI(
    title="Home Price Prediction API",
    description="Predict house prices with zipcode data",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    """Request model for full prediction with zipcode data"""
    features: Dict[str, Any]
    version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

class MinimalPredictionRequest(BaseModel):
    """Request model for minimal prediction - all features must be provided"""
    features: Dict[str, Any]

class PredictionResponse(BaseModel):
    """Response model for predictions"""
    prediction: float
    model_version: str
    metadata: Optional[Dict[str, Any]] = None

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_loaded(),
        "model_version": model_manager.get_version(),
        "demographic_data_ready": featurizer.is_ready()
    }


class BulkPredictionRequest(BaseModel):
    """Request model for bulk predictions"""
    features: List[Dict[str, Any]]
    version: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

  

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Full prediction endpoint - enriches input with demographic data
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")

    if not featurizer.is_ready():
        raise HTTPException(status_code=503, detail="Data enricher not ready")
    
    # Convert request to dict
    input = request.model_dump()
    logger.info(f"Received prediction request: {input}")
    features = input.get("features", {})
    
    # Enrich with demographic data
    try:
        enriched_features = await featurizer.enrich(features)
        # logger.info(f"Enriched features: {enriched_features}")
    except Exception as e:
        logger.error(f"Failed to enrich features: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to enrich data: {str(e)}")
    
    # Make prediction
    try:
        prediction = model_manager.predict(enriched_features)
        return PredictionResponse(
            prediction=prediction,
            model_version=model_manager.get_version(),
            metadata={"enriched": True}
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.post("/predict-strict", response_model=PredictionResponse)
async def predict_strict(request: MinimalPredictionRequest):
    """
    Strict prediction endpoint - expects *only* the required features from client    
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Validate that all required features are present
    required_features = model_manager.get_required_features()
    provided_features = set(request.features.keys())
    missing_features = required_features - provided_features
    excess_features = provided_features - required_features
    
    if missing_features or excess_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {list(missing_features)}; Excess features: {list(excess_features)}    "
        )
    
    # Make prediction
    try:
        prediction = model_manager.predict(request.features)
        return PredictionResponse(
            prediction=prediction,
            model_version=model_manager.get_version(),
            metadata={"enriched": False}
        )
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=400, detail=f"Prediction failed: {str(e)}")

@app.get("/model/versions")
async def list_model_versions():
    """
    List available model versions
    """
    versions = model_manager.list_versions()
    return {
        "available_versions": versions,
        "current_version": model_manager.get_version()
    }

class ModelReloadRequest(BaseModel):
    version: Optional[str] = None

class ModelReloadResponse(BaseModel):
    status: str
    message: str
    version: str

@app.post("/model/reload", response_model=ModelReloadResponse)
async def reload_model(request: ModelReloadRequest):
    """
    Reload the model (useful for hot-swapping new versions)
    """
    success = model_manager.reload(request.version)
    if success:
        return {
            "status": "success",
            "message": "Model reloaded",
            "version": model_manager.get_version()
        }
    else:
        raise HTTPException(status_code=500, detail="Failed to reload model")

@app.get("/model/info")
async def model_info():
    """
    Get current model information
    """
    return {
        "loaded": model_manager.is_loaded(),
        "version": model_manager.get_version(),
        "features": list(model_manager.get_required_features()),
        "feature_count": len(model_manager.get_required_features())
    }