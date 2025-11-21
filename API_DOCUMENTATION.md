# 📡 API Documentation

Complete reference for the Breast Histopathology AI REST API.

---

## Base URL

- **Local Development**: `http://localhost:8000`
- **Production**: `https://your-deployment-url.railway.app`

---

## Authentication

Currently, the API does not require authentication. For production deployment, consider implementing:
- API Keys
- OAuth2
- JWT tokens

---

## Endpoints

### 1. Health Check

**`GET /health`**

Check if the API is running and the model is loaded.

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-21T15:30:45.123456"
}
```

**Status Codes**:
- `200 OK`: Service is healthy
- `503 Service Unavailable`: Service is down or model not loaded

**cURL Example**:
```bash
curl http://localhost:8000/health
```

---

### 2. Single Image Prediction

**`POST /predict/single`**

Analyze a single histopathology image and return diagnosis.

**Request**:

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `file` (required): Image file (PNG, JPG, JPEG)

**Response**:
```json
{
  "prediction": "benign",
  "confidence": 0.8734215,
  "probabilities": {
    "benign": 0.8734215,
    "malignant": 0.1265785
  },
  "num_patches": 42,
  "patch_breakdown": {
    "benign_patches": 35,
    "malignant_patches": 7
  },
  "processing_time_seconds": 2.34
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `prediction` | string | Final diagnosis: "benign" or "malignant" |
| `confidence` | float | Confidence score (0-1) for the prediction |
| `probabilities` | object | Probability for each class |
| `num_patches` | integer | Total number of patches analyzed |
| `patch_breakdown` | object | Count of patches per class |
| `processing_time_seconds` | float | Time taken for inference |

**Status Codes**:
- `200 OK`: Successful prediction
- `400 Bad Request`: Invalid file or format
- `500 Internal Server Error`: Processing error

**cURL Example**:
```bash
curl -X POST http://localhost:8000/predict/single \
  -F "file=@/path/to/image.png"
```

**Python Example**:
```python
import requests

url = "http://localhost:8000/predict/single"
files = {"file": open("image.png", "rb")}

response = requests.post(url, files=files)
result = response.json()

print(f"Diagnosis: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

**JavaScript Example**:
```javascript
const formData = new FormData();
formData.append('file', fileInput.files[0]);

fetch('http://localhost:8000/predict/single', {
    method: 'POST',
    body: formData
})
.then(response => response.json())
.then(data => {
    console.log('Prediction:', data.prediction);
    console.log('Confidence:', data.confidence);
});
```

---

### 3. Multiple Images Prediction

**`POST /predict/folder`**

Analyze multiple images and return aggregated diagnosis.

**Request**:

- **Content-Type**: `multipart/form-data`
- **Parameters**:
  - `files` (required): Multiple image files

**Response**:
```json
{
  "aggregated_diagnosis": {
    "prediction": "benign",
    "confidence": 0.6842,
    "probabilities": {
      "benign": 0.6842,
      "malignant": 0.3158
    }
  },
  "num_images": 14,
  "image_breakdown": {
    "benign_images": 10,
    "malignant_images": 4
  },
  "individual_predictions": [
    {
      "filename": "image1.png",
      "prediction": "benign",
      "confidence": 0.92,
      "probabilities": {
        "benign": 0.92,
        "malignant": 0.08
      }
    },
    ...
  ],
  "processing_time_seconds": 18.75
}
```

**Response Fields**:

| Field | Type | Description |
|-------|------|-------------|
| `aggregated_diagnosis` | object | Final diagnosis across all images |
| `num_images` | integer | Total number of images analyzed |
| `image_breakdown` | object | Count of images per class |
| `individual_predictions` | array | Results for each image |
| `processing_time_seconds` | float | Total processing time |

**Status Codes**:
- `200 OK`: Successful prediction
- `400 Bad Request`: No files uploaded or invalid format
- `500 Internal Server Error`: Processing error

**cURL Example**:
```bash
curl -X POST http://localhost:8000/predict/folder \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  -F "files=@image3.png"
```

**Python Example**:
```python
import requests
from pathlib import Path

url = "http://localhost:8000/predict/folder"

# Upload multiple files
files = [
    ('files', open('image1.png', 'rb')),
    ('files', open('image2.png', 'rb')),
    ('files', open('image3.png', 'rb'))
]

response = requests.post(url, files=files)
result = response.json()

print(f"Overall Diagnosis: {result['aggregated_diagnosis']['prediction']}")
print(f"Confidence: {result['aggregated_diagnosis']['confidence']:.2%}")
print(f"Images Analyzed: {result['num_images']}")

# Close files
for _, f in files:
    f.close()
```

---

### 4. API Documentation (Interactive)

**`GET /docs`**

Access the interactive Swagger UI documentation.

**URL**: http://localhost:8000/docs

**Features**:
- Interactive API testing
- Request/response examples
- Schema validation
- Try-it-out functionality

---

### 5. Alternative Documentation

**`GET /redoc`**

Access ReDoc documentation (alternative UI).

**URL**: http://localhost:8000/redoc

---

## Error Handling

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

| Status Code | Error | Solution |
|-------------|-------|----------|
| `400` | No file uploaded | Include file in request |
| `400` | Invalid file format | Use PNG, JPG, or JPEG |
| `415` | Unsupported media type | Check Content-Type header |
| `422` | Validation error | Check request format |
| `500` | Model inference error | Check server logs |
| `503` | Service unavailable | Server is starting up |

---

## Rate Limiting

Currently, no rate limiting is implemented. For production:

**Recommended Limits**:
- 100 requests per minute per IP
- 1000 requests per day per IP

**Implementation** (using FastAPI middleware):
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.post("/predict/single")
@limiter.limit("100/minute")
async def predict_single(request: Request, ...):
    ...
```

---

## CORS Configuration

**Current Settings**:
```python
allow_origins=["*"]  # All origins allowed
allow_methods=["*"]  # All methods allowed
allow_headers=["*"]  # All headers allowed
```

**Production Recommendation**:
```python
allow_origins=[
    "https://yourdomain.com",
    "https://app.yourdomain.com"
]
allow_methods=["GET", "POST"]
allow_headers=["Content-Type", "Authorization"]
```

---

## Performance Tips

### 1. Batch Processing
Instead of multiple single requests, use `/predict/folder` for better performance:

**Bad** (slow):
```python
for image in images:
    response = requests.post('/predict/single', files={'file': open(image, 'rb')})
```

**Good** (fast):
```python
files = [('files', open(img, 'rb')) for img in images]
response = requests.post('/predict/folder', files=files)
```

### 2. Async Requests
Use async HTTP clients for parallel requests:

```python
import aiohttp
import asyncio

async def predict_async(session, image_path):
    async with session.post(
        'http://localhost:8000/predict/single',
        data={'file': open(image_path, 'rb')}
    ) as response:
        return await response.json()

async def main():
    async with aiohttp.ClientSession() as session:
        tasks = [predict_async(session, img) for img in images]
        results = await asyncio.gather(*tasks)
        return results
```

### 3. Compression
Enable gzip compression in requests:

```python
response = requests.post(
    url,
    files=files,
    headers={'Accept-Encoding': 'gzip'}
)
```

---

## WebSocket Support (Future)

For real-time streaming results, WebSocket support could be added:

```python
@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_bytes()
        result = await process_image(data)
        await websocket.send_json(result)
```

---

## Client Libraries

### Python Client

```python
class BreastHistopathologyClient:
    def __init__(self, base_url):
        self.base_url = base_url
        
    def predict_single(self, image_path):
        with open(image_path, 'rb') as f:
            response = requests.post(
                f"{self.base_url}/predict/single",
                files={'file': f}
            )
        return response.json()
    
    def predict_batch(self, image_paths):
        files = [('files', open(img, 'rb')) for img in image_paths]
        response = requests.post(
            f"{self.base_url}/predict/folder",
            files=files
        )
        for _, f in files:
            f.close()
        return response.json()

# Usage
client = BreastHistopathologyClient("http://localhost:8000")
result = client.predict_single("image.png")
```

---

## Testing

### Unit Tests

```python
from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_predict_single():
    with open("test_image.png", "rb") as f:
        response = client.post(
            "/predict/single",
            files={"file": f}
        )
    assert response.status_code == 200
    assert "prediction" in response.json()
```

---

## Monitoring

### Health Checks

Regular health check pings:
```bash
# Every 30 seconds
watch -n 30 'curl http://localhost:8000/health'
```

### Logging

Enable detailed logging:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
logger.info(f"Prediction request received for {filename}")
```

---

## Security Best Practices

1. **Input Validation**: Always validate file types and sizes
2. **Rate Limiting**: Implement rate limits in production
3. **HTTPS**: Use HTTPS in production
4. **CORS**: Restrict origins to known domains
5. **Authentication**: Add API keys or OAuth
6. **File Size Limits**: Limit max upload size (default: 10MB)
7. **Sanitization**: Sanitize file names to prevent path traversal

---

## Support

For API-related issues:
- Check `/docs` for interactive testing
- Review server logs for errors
- Submit issues on GitHub

---

**Last Updated**: November 2025
**API Version**: 1.0.0

