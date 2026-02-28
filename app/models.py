from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    features: List[float] = Field(
        ...,
        description="List of numerical input features for the model.",
        example=[0.5, 1.2, 3.4, 2.1]
    )


class PredictionResponse(BaseModel):
    label: str = Field(..., description="Predicted class label.", example="positive")
    confidence: float = Field(..., description="Model confidence score between 0 and 1.", example=0.87)
    model_version: str = Field(..., description="Version of the model used.", example="v1")


class HealthResponse(BaseModel):
    status: str = Field(..., example="ok")
    model_version: str = Field(..., example="v1")
    