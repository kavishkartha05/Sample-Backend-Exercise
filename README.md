# Sample Prediction API

Built to demonstrate RESTful API design, Pydantic schema validation, and unit testing with pytest.

---

## Project Structure

```
fastapi_sample/
  app/
    main.py        # FastAPI app and route definitions
    models.py      # Pydantic request/response schemas
    predictor.py   # Model inference logic
  tests/
    test_main.py   # Unit tests (pytest + FastAPI TestClient)
  requirements.txt
  README.md
```

---

## Quickstart

**1. Install dependencies**
```bash
pip install -r requirements.txt
```

**2. Run the server**
```bash
uvicorn app.main:app --reload
```

**3. Check health**
```
GET http://127.0.0.1:8000/health
```
```json
{ "status": "ok", "model_version": "v1" }
```

**4. Run a prediction**
```
POST http://127.0.0.1:8000/predict
Content-Type: application/json

{ "features": [0.5, 1.2, 3.4, 2.1] }
```
```json
{
  "label": "positive",
  "confidence": 0.73,
  "model_version": "v1"
}
```

---

## API Endpoints

| Method | Endpoint   | Description                        |
|--------|------------|------------------------------------|
| GET    | /health    | Health check, returns model version |
| POST   | /predict   | Accepts features, returns prediction |

**HTTP status codes used:**
- `200` — success
- `400` — empty features list
- `422` — malformed request body (FastAPI validation)

---

## Running Tests

```bash
pytest tests/ -v
```

Tests:
- Health endpoint (should return 200 with expected fields)
- Predict endpoint response contract (should return label, confidence, model_version)
- Empty features (should return 400)
- Missing body (should return 422)
- Confidence score (should always clamped between 0 and 1)
- Label (should map to positive or negative based on threshold)

---
