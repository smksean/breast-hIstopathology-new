# 🚀 Deployment Guide - Breast Histopathology Classification

This guide provides step-by-step instructions for deploying the Breast Histopathology Classification system in various environments.

## 📋 Table of Contents

1. [Prerequisites](#prerequisites)
2. [Model Export](#model-export)
3. [Local Deployment](#local-deployment)
4. [Docker Deployment](#docker-deployment)
5. [Cloud Deployment](#cloud-deployment)
6. [Production Considerations](#production-considerations)
7. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### System Requirements
- **OS**: Windows 10+, Ubuntu 18.04+, macOS 10.14+
- **RAM**: Minimum 4GB, Recommended 8GB+
- **Disk Space**: Minimum 2GB free
- **GPU** (Optional): CUDA-compatible GPU for faster inference

### Software Requirements
- Python 3.10 or higher
- pip (Python package manager)
- Docker (for containerized deployment)
- Git

---

## Model Export

### Step 1: Locate Your Trained Model

If you trained the model in Google Colab:
```python
# In Colab, your best model is at:
/content/drive/MyDrive/BreakHis/checkpoints/best_model.pth
```

### Step 2: Download the Model

**Option A: From Google Drive**
```bash
# Download from Google Drive to local machine
# Use the Drive web interface or gdown
pip install gdown
gdown --id YOUR_FILE_ID -O best_model.pth
```

**Option B: Direct from Colab**
```python
# In Colab
from google.colab import files
files.download('/content/drive/MyDrive/BreakHis/checkpoints/best_model.pth')
```

### Step 3: Export Model for Production

```bash
# Place the model in your project directory
cd breast-histopathology

# Run the export script
python model_export.py /path/to/best_model.pth
```

**Expected Output**:
```
🔧 Initializing model architecture...
📥 Loading weights from /path/to/best_model.pth...
✅ Loaded checkpoint from epoch 4
   Validation accuracy: 90.55%
💾 Saved model state dict: deployment/models/breast_histopathology_resnet50.pth
📄 Saved model metadata: deployment/models/model_metadata.json
🚀 Exporting to TorchScript format...
✅ Saved TorchScript model: deployment/models/breast_histopathology_resnet50_traced.pt
```

### Step 4: Verify Export

```bash
# Check exported files
ls -lh deployment/models/
```

You should see:
- `breast_histopathology_resnet50.pth` (~90MB)
- `model_metadata.json` (~1KB)
- `breast_histopathology_resnet50_traced.pt` (~90MB)

---

## Local Deployment

### Step 1: Install Dependencies

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Step 2: Verify Installation

```bash
# Test the inference pipeline
python inference_pipeline.py \
    deployment/models/breast_histopathology_resnet50.pth \
    deployment/models/model_metadata.json \
    /path/to/test/image.png
```

### Step 3: Start the API Server

```bash
# Production mode
python app.py --host 0.0.0.0 --port 8000

# Development mode (with auto-reload)
python app.py --reload
```

**Console Output**:
```
🚀 Starting Breast Histopathology Classification API
═══════════════════════════════════════════════════
🖥️  Using device: cuda
📄 Loaded metadata: model_metadata.json
✅ Model loaded: breast_histopathology_resnet50.pth
═══════════════════════════════════════════════════

🚀 Starting server at http://0.0.0.0:8000
📖 API documentation at http://0.0.0.0:8000/docs
```

### Step 4: Test the API

```bash
# Health check
curl http://localhost:8000/health

# Single image prediction
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.png"
```

### Step 5: Access Web Interface

```bash
# Serve the frontend
cd templates
python -m http.server 8080
```

Visit: http://localhost:8080

---

## Docker Deployment

### Step 1: Build Docker Image

```bash
# Build the image
docker build -t breast-histopathology:latest .

# Verify build
docker images | grep breast-histopathology
```

### Step 2: Run Container

```bash
# Run with model mounted from host
docker run -d \
  --name breast-histopathology-api \
  -p 8000:8000 \
  -v $(pwd)/deployment/models:/app/deployment/models:ro \
  breast-histopathology:latest

# Check logs
docker logs -f breast-histopathology-api
```

### Step 3: Docker Compose (Recommended)

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Stop services
docker-compose down
```

### Step 4: Verify Deployment

```bash
# Health check
curl http://localhost:8000/health

# Or open in browser
open http://localhost:8000/docs
```

---

## Cloud Deployment

### AWS Deployment (EC2)

#### 1. Launch EC2 Instance

```bash
# Recommended instance type:
# - t2.medium (2 vCPU, 4GB RAM) - minimum
# - g4dn.xlarge (4 vCPU, 16GB RAM, 1 GPU) - optimal

# AMI: Ubuntu Server 20.04 LTS
# Security Group: Open ports 22, 80, 8000
```

#### 2. Connect and Setup

```bash
# SSH into instance
ssh -i your-key.pem ubuntu@your-ec2-ip

# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose
```

#### 3. Deploy Application

```bash
# Clone repository
git clone https://github.com/yourusername/breast-histopathology.git
cd breast-histopathology

# Copy model files (from S3 or your local machine)
# Using SCP from local:
scp -i your-key.pem deployment/models/* ubuntu@your-ec2-ip:~/breast-histopathology/deployment/models/

# Start with Docker Compose
docker-compose up -d

# Verify
curl http://localhost:8000/health
```

#### 4. Configure Domain (Optional)

```bash
# Install Nginx
sudo apt install nginx -y

# Configure reverse proxy
sudo nano /etc/nginx/sites-available/breast-histopathology

# Add configuration:
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}

# Enable site
sudo ln -s /etc/nginx/sites-available/breast-histopathology /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### GCP Deployment (Cloud Run)

#### 1. Prepare for Cloud Run

```bash
# Install Google Cloud SDK
curl https://sdk.cloud.google.com | bash
gcloud init

# Enable required APIs
gcloud services enable run.googleapis.com
gcloud services enable containerregistry.googleapis.com
```

#### 2. Build and Push to Container Registry

```bash
# Set project
export PROJECT_ID=your-project-id

# Build for Cloud Run
docker build -t gcr.io/$PROJECT_ID/breast-histopathology:latest .

# Push to GCR
docker push gcr.io/$PROJECT_ID/breast-histopathology:latest
```

#### 3. Deploy to Cloud Run

```bash
gcloud run deploy breast-histopathology \
  --image gcr.io/$PROJECT_ID/breast-histopathology:latest \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated \
  --memory 4Gi \
  --cpu 2
```

### Azure Deployment (Container Instances)

```bash
# Login to Azure
az login

# Create resource group
az group create --name breast-histopathology-rg --location eastus

# Push to Azure Container Registry
az acr create --resource-group breast-histopathology-rg \
  --name yourregistry --sku Basic

az acr build --registry yourregistry \
  --image breast-histopathology:latest .

# Deploy container
az container create \
  --resource-group breast-histopathology-rg \
  --name breast-histopathology \
  --image yourregistry.azurecr.io/breast-histopathology:latest \
  --cpu 2 --memory 4 \
  --dns-name-label breast-histopathology-api \
  --ports 8000
```

---

## Production Considerations

### 1. Security

```bash
# Enable HTTPS with Let's Encrypt
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d your-domain.com
```

```python
# Add authentication to API (app.py)
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

@app.post("/predict")
async def predict(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    file: UploadFile = File(...)
):
    # Verify token
    if credentials.credentials != "your-secret-token":
        raise HTTPException(status_code=401, detail="Invalid token")
    # ... rest of code
```

### 2. Monitoring

```python
# Add logging
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/predict")
async def predict(...):
    logger.info(f"Prediction request received: {file.filename}")
    # ... rest of code
```

### 3. Rate Limiting

```python
# Install slowapi
pip install slowapi

from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(429, _rate_limit_exceeded_handler)

@app.post("/predict")
@limiter.limit("10/minute")
async def predict(...):
    # ... code
```

### 4. Scaling

**Horizontal Scaling with Load Balancer**:

```yaml
# docker-compose.yml for scaling
version: '3.8'
services:
  api:
    image: breast-histopathology:latest
    deploy:
      replicas: 3  # Run 3 instances
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
  
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    depends_on:
      - api
```

```bash
# Scale with docker-compose
docker-compose up -d --scale api=3
```

### 5. Backup and Recovery

```bash
# Automated model backup script
#!/bin/bash
# backup_models.sh

BACKUP_DIR="/backups/models"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup models
cp -r deployment/models $BACKUP_DIR/models_$DATE

# Keep only last 7 days
find $BACKUP_DIR -type d -mtime +7 -exec rm -rf {} +

# Upload to S3 (optional)
aws s3 sync $BACKUP_DIR s3://your-bucket/backups/
```

```bash
# Add to crontab
crontab -e
# Add line:
0 2 * * * /path/to/backup_models.sh
```

---

## Troubleshooting

### Issue 1: Model Not Loading

**Error**: `Model file not found`

**Solution**:
```bash
# Verify model path
ls -la deployment/models/

# Check permissions
chmod 644 deployment/models/*.pth

# Verify model is not corrupted
python -c "import torch; torch.load('deployment/models/breast_histopathology_resnet50.pth')"
```

### Issue 2: Out of Memory

**Error**: `RuntimeError: CUDA out of memory`

**Solution**:
```python
# Reduce batch size in inference_pipeline.py
# Or use CPU inference
pipeline = InferencePipeline(..., device='cpu')
```

### Issue 3: Slow Inference

**Optimization**:
```python
# Enable TorchScript
model = torch.jit.load('deployment/models/breast_histopathology_resnet50_traced.pt')

# Use GPU
model = model.to('cuda')

# Enable mixed precision (if GPU supports it)
model = model.half()  # FP16
```

### Issue 4: Port Already in Use

**Error**: `OSError: [Errno 48] Address already in use`

**Solution**:
```bash
# Find process using port 8000
lsof -i :8000

# Kill process
kill -9 <PID>

# Or use different port
python app.py --port 8001
```

### Issue 5: CORS Errors in Frontend

**Solution**:
```python
# Update CORS settings in app.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "https://your-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## Health Monitoring

### Setup Health Checks

```python
# Advanced health check endpoint
@app.get("/health/detailed")
async def detailed_health():
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": {
            "loaded": pipeline is not None,
            "device": str(pipeline.device) if pipeline else None,
            "memory_used": torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        },
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    }
```

### Monitoring Script

```bash
#!/bin/bash
# monitor.sh - Check API health every 5 minutes

while true; do
    response=$(curl -s http://localhost:8000/health)
    status=$(echo $response | jq -r '.status')
    
    if [ "$status" != "healthy" ]; then
        echo "$(date): API unhealthy! Sending alert..."
        # Send alert (email, Slack, etc.)
    fi
    
    sleep 300
done
```

---

## Performance Tuning

### 1. Enable Model Quantization

```python
# Quantize model for faster inference
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)
```

### 2. Batch Processing

```python
# Process multiple patches in parallel
def predict_batched(patches, batch_size=32):
    results = []
    for i in range(0, len(patches), batch_size):
        batch = patches[i:i+batch_size]
        # Process batch
        results.extend(model(batch))
    return results
```

### 3. Caching

```python
# Cache predictions (for demo/testing)
from functools import lru_cache
import hashlib

@lru_cache(maxsize=100)
def cached_predict(image_hash):
    # Predict and cache
    pass
```

---

## Conclusion

You now have a complete guide for deploying the Breast Histopathology Classification system!

**Next Steps**:
1. ✅ Export your trained model
2. ✅ Test locally
3. ✅ Deploy to cloud
4. ✅ Set up monitoring
5. ✅ Optimize performance

For questions or issues, please open an issue on GitHub.

---

**Last Updated**: November 19, 2025




