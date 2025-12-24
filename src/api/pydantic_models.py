# src/api/pydantic_models.py
from pydantic import BaseModel, Field
from typing import List, Optional

class CustomerFeatures(BaseModel):
    recency: float = Field(..., ge=0, description="Days since last transaction")
    frequency: float = Field(..., gt=0, description="Number of transactions")
    total_amount: float = Field(..., description="Total transaction amount")
    avg_amount: float = Field(..., gt=0, description="Average transaction amount")
    std_amount: float = Field(..., ge=0, description="Standard deviation of amounts")
    amount_variability: float = Field(..., ge=0, description="Amount variability (std/mean)")
    amount_range: float = Field(..., ge=0, description="Range of transaction amounts")

class PredictionRequest(BaseModel):
    customer_id: Optional[str] = Field(None, description="Customer identifier")
    features: CustomerFeatures

class BatchPredictionRequest(BaseModel):
    customers: List[CustomerFeatures]

class PredictionResponse(BaseModel):
    customer_id: Optional[str]
    risk_probability: float = Field(..., ge=0, le=1, description="Probability of being high-risk")
    risk_class: int = Field(..., ge=0, le=1, description="0=Low Risk, 1=High Risk")
    risk_level: str
    features_used: List[str]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    api_version: str = "1.0.0"

class ModelInfo(BaseModel):
    model_type: str
    model_path: str
    n_features: int
    feature_names: List[str]
    api_endpoints: List[str] = ["/predict", "/predict-batch", "/health", "/model-info"]