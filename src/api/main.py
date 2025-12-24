# src/api/main.py - FASTAPI APPLICATION
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
import pandas as pd
from typing import List, Optional
import os

app = FastAPI(
    title="Credit Risk Prediction API",
    description="API for predicting credit risk using RFM features",
    version="1.0.0"
)

# Load model and scaler
MODEL_PATH = "models/xgboost_final.pkl"  # Update with your best model
SCALER_PATH = "models/scaler.pkl"

try:
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print(f"✅ Model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None
    scaler = None

# Define feature names (update based on your actual features)
FEATURE_NAMES = [
    'recency', 'frequency', 'total_amount', 'avg_amount', 
    'std_amount', 'amount_variability', 'amount_range'
]

# Pydantic models for request/response
class CustomerFeatures(BaseModel):
    """Input features for a customer"""
    recency: float
    frequency: float
    total_amount: float
    avg_amount: float
    std_amount: float
    amount_variability: float
    amount_range: float
    
    class Config:
        schema_extra = {
            "example": {
                "recency": 45.5,
                "frequency": 12.0,
                "total_amount": 50000.0,
                "avg_amount": 4166.67,
                "std_amount": 1500.0,
                "amount_variability": 0.36,
                "amount_range": 3000.0
            }
        }

class PredictionResponse(BaseModel):
    """API response format"""
    customer_id: Optional[str] = None
    risk_probability: float
    risk_class: int
    risk_level: str
    features_used: List[str]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    model_loaded: bool
    model_type: str

# API Endpoints
@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    return {
        "status": "Credit Risk API is running",
        "model_loaded": model is not None,
        "model_type": "XGBoost" if model is not None else "None"
    }

@app.get("/health")
async def health_check():
    """Simple health check"""
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
async def predict(features: CustomerFeatures, customer_id: Optional[str] = None):
    """
    Predict credit risk for a customer
    
    - **features**: Customer's RFM features
    - **customer_id**: Optional customer identifier
    """
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Convert features to numpy array
        feature_values = np.array([[
            features.recency,
            features.frequency,
            features.total_amount,
            features.avg_amount,
            features.std_amount,
            features.amount_variability,
            features.amount_range
        ]])
        
        # Scale features
        features_scaled = scaler.transform(feature_values)
        
        # Make prediction
        risk_probability = float(model.predict_proba(features_scaled)[0, 1])
        risk_class = 1 if risk_probability > 0.5 else 0
        risk_level = "High Risk" if risk_class == 1 else "Low Risk"
        
        return {
            "customer_id": customer_id,
            "risk_probability": round(risk_probability, 4),
            "risk_class": risk_class,
            "risk_level": risk_level,
            "features_used": FEATURE_NAMES
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {str(e)}")

@app.post("/predict-batch")
async def predict_batch(customers: List[CustomerFeatures]):
    """Predict for multiple customers at once"""
    if model is None or scaler is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        results = []
        for i, customer in enumerate(customers):
            feature_values = np.array([[
                customer.recency,
                customer.frequency,
                customer.total_amount,
                customer.avg_amount,
                customer.std_amount,
                customer.amount_variability,
                customer.amount_range
            ]])
            
            features_scaled = scaler.transform(feature_values)
            risk_probability = float(model.predict_proba(features_scaled)[0, 1])
            risk_class = 1 if risk_probability > 0.5 else 0
            
            results.append({
                "customer_index": i,
                "risk_probability": round(risk_probability, 4),
                "risk_class": risk_class,
                "risk_level": "High Risk" if risk_class == 1 else "Low Risk"
            })
        
        return {
            "predictions": results,
            "total_customers": len(results),
            "high_risk_count": sum(r["risk_class"] for r in results)
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Batch prediction error: {str(e)}")

@app.get("/model-info")
async def model_info():
    """Get information about the loaded model"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    model_type = type(model).__name__
    if hasattr(model, 'n_estimators'):
        n_estimators = model.n_estimators
    else:
        n_estimators = "N/A"
    
    return {
        "model_type": model_type,
        "model_path": MODEL_PATH,
        "n_features": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "n_estimators": n_estimators
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)