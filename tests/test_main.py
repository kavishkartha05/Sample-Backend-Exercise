import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.predictor import run_prediction, _mock_inference

client = TestClient(app)


# health check endpoint

def test_health_returns_200():
    response = client.get("/health")
    assert response.status_code == 200


def test_health_returns_expected_fields():
    response = client.get("/health")
    data = response.json()
    assert data["status"] == "ok"
    assert "model_version" in data


# predictor endpoint tests

def test_predict_returns_200():
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    assert response.status_code == 200


def test_predict_response_contract():
    """Ensure the response always contains the expected fields with valid types."""
    response = client.post("/predict", json={"features": [1.0, 2.0, 3.0]})
    data = response.json()

    assert "label" in data
    assert "confidence" in data
    assert "model_version" in data
    assert data["label"] in ["positive", "negative"]
    assert 0.0 <= data["confidence"] <= 1.0


def test_predict_empty_features_returns_400():
    """Empty features list should raise a 400 Bad Request."""
    response = client.post("/predict", json={"features": []})
    assert response.status_code == 400


def test_predict_missing_features_returns_422():
    """Missing required field should return 422 Unprocessable Entity (FastAPI validation)."""
    response = client.post("/predict", json={})
    assert response.status_code == 422

# predictor logic tests

def test_confidence_clamped_between_0_and_1():
    """Confidence score should always be within valid probability range."""
    confidence = _mock_inference([999.9, 999.9, 999.9])
    assert 0.0 <= confidence <= 1.0


def test_positive_label_above_threshold():
    result = run_prediction([5.0, 5.0, 5.0])
    assert result.label == "positive"
    assert result.confidence >= 0.5


def test_negative_label_below_threshold():
    result = run_prediction([0.1, 0.1, 0.1])
    assert result.label == "negative"
    assert result.confidence < 0.5


def test_model_version_present():
    result = run_prediction([1.0, 2.0])
    assert result.model_version == "v1"
    