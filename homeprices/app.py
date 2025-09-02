from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any, Optional
import os
from contextlib import asynccontextmanager
from model_manager import ModelManager
from data_enricher import DataEnricher
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global instances
model_manager = ModelManager()
data_enricher = DataEnricher()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    # Load model on startup
    model_loaded = model_manager.load_model()
    if model_loaded:
        logger.info(f"Model loaded successfully: {model_manager.get_version()}")
    else:
        logger.warning("No model loaded - predictions will fail until model is available")
    
    # Initialize data enricher
    await data_enricher.initialize()
    
    yield
    
    # Cleanup
    await data_enricher.cleanup()

app = FastAPI(
    title="Home Price Prediction API",
    description="Predict house prices with demographic enrichment",
    version="1.0.0",
    lifespan=lifespan
)

class PredictionRequest(BaseModel):
    """Request model for full prediction with zipcode enrichment"""
    bedrooms: float
    bathrooms: float
    sqft_living: float
    sqft_lot: float
    floors: float
    waterfront: float
    view: float
    condition: float
    grade: float
    sqft_above: float
    sqft_basement: float
    yr_built: float
    yr_renovated: float
    zipcode: str
    lat: float
    long: float
    sqft_living15: float
    sqft_lot15: float

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
        "data_enricher_ready": data_enricher.is_ready()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Full prediction endpoint - enriches input with demographic data
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Convert request to dict
    features = request.model_dump()
    
    # Enrich with demographic data
    try:
        enriched_features = await data_enricher.enrich(features)
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

@app.post("/predict-minimal", response_model=PredictionResponse)
async def predict_minimal(request: MinimalPredictionRequest):
    """
    Minimal prediction endpoint - expects all required features from client
    No automatic enrichment is performed
    """
    if not model_manager.is_loaded():
        raise HTTPException(status_code=503, detail="Model not available")
    
    # Validate that all required features are present
    required_features = model_manager.get_required_features()
    provided_features = set(request.features.keys())
    missing_features = required_features - provided_features
    
    if missing_features:
        raise HTTPException(
            status_code=400,
            detail=f"Missing required features: {list(missing_features)}"
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

@app.post("/model/reload")
async def reload_model():
    """
    Reload the model (useful for hot-swapping new versions)
    """
    success = model_manager.reload()
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