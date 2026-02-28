from fastapi import FastAPI, HTTPException
from app.models import PredictionRequest, PredictionResponse, HealthResponse
from app.predictor import run_prediction

app = FastAPI(
    title="Prediction API",
    description="A sample REST API for running AI model predictions.",
    version="1.0.0"
)


@app.get("/health", response_model=HealthResponse)
def health_check():
    return HealthResponse(status="ok", model_version="v1")


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if not request.features:
        raise HTTPException(status_code=400, detail="Features list cannot be empty.")

    result = run_prediction(request.features)
    return result
