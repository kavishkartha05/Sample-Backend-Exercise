from app.models import PredictionResponse

MODEL_VERSION = "v1"

# defining confidence thresholds above which to classify as "positive"
THRESHOLD = 0.5


def _mock_inference(features: list[float]) -> float:
    if not features:
        return 0.0

    raw_score = sum(features) / (len(features) * 10)

    # for the purpose of this exercise: clamping to [0, 1] to simulate a probability output
    confidence = max(0.0, min(1.0, abs(raw_score)))
    return round(confidence, 4)


def run_prediction(features: list[float]) -> PredictionResponse:
    confidence = _mock_inference(features)
    label = "positive" if confidence >= THRESHOLD else "negative"

    return PredictionResponse(
        label=label,
        confidence=confidence,
        model_version=MODEL_VERSION)
