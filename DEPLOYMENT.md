# 🚀 Deployment Guide

Complete guide for deploying the Breast Histopathology AI system to production.

---

## 📑 Table of Contents

- [Prerequisites](#prerequisites)
- [Docker Deployment](#docker-deployment)
- [Cloud Deployment](#cloud-deployment)
  - [Railway.app](#1-railwayapp-recommended)
  - [Render.com](#2-rendercom)
  - [Fly.io](#3-flyio)
  - [Hugging Face Spaces](#4-hugging-face-spaces)
- [Post-Deployment](#post-deployment)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

---

## Prerequisites

### Required

- ✅ Git installed
- ✅ GitHub account
- ✅ Code pushed to GitHub repository
- ✅ Docker installed (for local testing)

### Optional

- Docker Desktop (for local testing)
- Cloud platform account (Railway, Render, etc.)

---

## Docker Deployment

### Local Docker Setup

**1. Build the image**:
```bash
docker build -t breast-histopathology-ai .
```

**2. Run the container**:
```bash
docker run -d \
  -p 8000:8000 \
  --name breast-ai \
  breast-histopathology-ai
```

**3. Test the deployment**:
```bash
# Health check
curl http://localhost:8000/health

# API documentation
open http://localhost:8000/docs
```

**4. View logs**:
```bash
docker logs breast-ai
```

**5. Stop the container**:
```bash
docker stop breast-ai
docker rm breast-ai
```

### Docker Compose (Alternative)

Create `docker-compose.yml`:
```yaml
version: '3.8'

services:
  api:
    build: .
    ports:
      - "8000:8000"
    environment:
      - MODEL_PATH=/app/models/best_model.pth
    volumes:
      - ./models:/app/models:ro
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

**Start**:
```bash
docker-compose up -d
```

---

## Cloud Deployment

### 1. Railway.app (Recommended)

**Why Railway?**
- ✅ $5 free credit/month
- ✅ Automatic Docker deployment
- ✅ No credit card for trial
- ✅ GitHub integration
- ✅ Automatic HTTPS

#### Step-by-Step

**Step 1: Sign Up**
1. Go to [railway.app](https://railway.app)
2. Click "Start a New Project"
3. Sign in with GitHub

**Step 2: Create Project**
1. Click "Deploy from GitHub repo"
2. Select: `smksean/breast-hIstopathology-new`
3. Click "Deploy Now"

**Step 3: Configure**
- Railway automatically detects `Dockerfile`
- Build starts immediately
- Wait 5-10 minutes for first deployment

**Step 4: Get Your URL**
1. Go to project Settings
2. Click "Generate Domain"
3. Copy your URL: `https://breast-histopathology-ai-production.up.railway.app`

**Step 5: Update Web Interface**

Edit `web/app.js` line 4:
```javascript
const API_URL = 'https://breast-histopathology-ai-production.up.railway.app';
```

Commit and push:
```bash
git add web/app.js
git commit -m "Update API URL for production"
git push
```

Railway auto-deploys the update!

#### Railway Configuration

**Environment Variables** (optional):
```bash
PORT=8000
PYTHON_VERSION=3.9
```

**Resource Limits**:
- Memory: 512MB (default)
- CPU: Shared
- Disk: 1GB ephemeral

#### Keeping Service Awake

Railway free tier sleeps after inactivity. To keep awake:

**Option 1: Cron Job**
```bash
# Use cron-job.org to ping every 5 minutes
curl https://your-app.railway.app/health
```

**Option 2: UptimeRobot**
1. Sign up at [uptimerobot.com](https://uptimerobot.com)
2. Add monitor: `https://your-app.railway.app/health`
3. Check interval: 5 minutes

---

### 2. Render.com

**Why Render?**
- ✅ 750 free hours/month
- ✅ Docker support
- ✅ Auto-deploy from GitHub
- ✅ Built-in SSL

#### Step-by-Step

**Step 1: Sign Up**
1. Go to [render.com](https://render.com)
2. Sign in with GitHub

**Step 2: Create Web Service**
1. Click "New +" → "Web Service"
2. Connect repository: `smksean/breast-hIstopathology-new`
3. Configure:
   ```
   Name: breast-histopathology-ai
   Environment: Docker
   Region: Oregon (US West)
   Branch: main
   Instance Type: Free
   ```

**Step 3: Advanced Settings**
```bash
# Health Check Path
/health

# Port (auto-detected)
8000
```

**Step 4: Deploy**
- Click "Create Web Service"
- Wait 10-15 minutes for first build
- Get URL: `https://breast-histopathology-ai.onrender.com`

#### Render Limitations

- **Cold Starts**: Service spins down after 15 min inactivity
- **Wake-up Time**: 30-60 seconds on first request
- **Build Time**: 10-15 minutes per deploy

---

### 3. Fly.io

**Why Fly.io?**
- ✅ Free tier includes 3 shared VMs
- ✅ Global edge network
- ✅ Fast deployments

#### Step-by-Step

**Step 1: Install Fly CLI**
```bash
# Windows
powershell -Command "iwr https://fly.io/install.ps1 -useb | iex"

# Mac/Linux
curl -L https://fly.io/install.sh | sh
```

**Step 2: Login**
```bash
fly auth login
```

**Step 3: Launch App**
```bash
fly launch

# Follow prompts:
# Name: breast-histopathology-ai
# Region: Choose closest
# Database: No
```

**Step 4: Deploy**
```bash
fly deploy
```

**Step 5: Get URL**
```bash
fly status
# URL: https://breast-histopathology-ai.fly.dev
```

#### Fly Configuration

Create `fly.toml` (auto-generated):
```toml
app = "breast-histopathology-ai"

[build]
  image = "breast-histopathology-ai:latest"

[http_service]
  internal_port = 8000
  force_https = true

[[services]]
  http_checks = []
  internal_port = 8000
  protocol = "tcp"

  [[services.ports]]
    handlers = ["http"]
    port = 80

  [[services.ports]]
    handlers = ["tls", "http"]
    port = 443
```

---

### 4. Hugging Face Spaces

**Why Hugging Face?**
- ✅ Free forever (community plan)
- ✅ Perfect for ML demos
- ✅ Built-in model hosting

#### Step-by-Step

**Step 1: Create Space**
1. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
2. Click "Create new Space"
3. Choose "Docker" as SDK

**Step 2: Push Code**
```bash
# Add HF as remote
git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/breast-histopathology-ai

# Push
git push hf main
```

**Step 3: Configure**
Create `README.md` header:
```yaml
---
title: Breast Histopathology AI
emoji: 🔬
colorFrom: pink
colorTo: red
sdk: docker
pinned: false
---
```

**Step 4: Access**
URL: `https://huggingface.co/spaces/YOUR_USERNAME/breast-histopathology-ai`

---

## Post-Deployment

### 1. Update Web Interface

**Edit `web/app.js` line 4**:
```javascript
const API_URL = 'https://your-production-url.com';
```

### 2. Test API Endpoints

```bash
# Health check
curl https://your-app-url.com/health

# Single prediction
curl -X POST https://your-app-url.com/predict/single \
  -F "file=@test_image.png"
```

### 3. Enable CORS for Production

**Edit `api.py`**:
```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://yourdomain.com",
        "http://localhost:3000"  # For development
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### 4. Set Up Monitoring

**Railway**:
- Built-in metrics dashboard
- View logs in real-time

**Render**:
- Metrics tab shows CPU/Memory
- Logs available in dashboard

**Custom Monitoring**:
```python
# Add to api.py
import time
from prometheus_client import Counter, Histogram

request_count = Counter('requests_total', 'Total requests')
request_duration = Histogram('request_duration_seconds', 'Request duration')

@app.middleware("http")
async def monitor_requests(request: Request, call_next):
    request_count.inc()
    start = time.time()
    response = await call_next(request)
    request_duration.observe(time.time() - start)
    return response
```

---

## Monitoring

### Health Checks

**Automated Monitoring**:
```bash
# Ping every 5 minutes
*/5 * * * * curl https://your-app.railway.app/health
```

**UptimeRobot Setup**:
1. Create account at [uptimerobot.com](https://uptimerobot.com)
2. Add monitor
3. Monitor Type: HTTP(s)
4. URL: `https://your-app.railway.app/health`
5. Monitoring Interval: 5 minutes
6. Alert Contacts: Your email

### Logging

**View Logs**:
```bash
# Railway
railway logs

# Render
# Use web dashboard

# Fly.io
fly logs
```

**Custom Logging**:
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

@app.post("/predict/single")
async def predict_single(file: UploadFile):
    logger.info(f"Prediction request: {file.filename}")
    # ... rest of code
```

---

## Troubleshooting

### Build Failures

**Problem**: `libgl1-mesa-glx` not found

**Solution**: Already fixed in Dockerfile (uses `libgl1` instead)

**Problem**: Out of memory during build

**Solution**: 
```dockerfile
# Use multi-stage build
FROM python:3.9-slim as builder
# ... pip install
FROM python:3.9-slim
COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
```

### Runtime Errors

**Problem**: Model file not found

**Solution**: Check `.dockerignore` doesn't exclude `models/`

**Problem**: Port binding error

**Solution**: Ensure `Dockerfile` exposes correct port:
```dockerfile
EXPOSE 8000
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Performance Issues

**Problem**: Slow cold starts

**Solutions**:
1. Use Railway/Render paid tier (always-on)
2. Set up ping service to keep warm
3. Optimize Docker image size

**Problem**: Out of memory

**Solutions**:
1. Reduce batch size in `predict.py`
2. Upgrade to paid tier with more RAM
3. Implement request queuing

---

## Performance Optimization

### 1. Docker Image Optimization

**Multi-stage build**:
```dockerfile
# Build stage
FROM python:3.9-slim as builder
WORKDIR /build
COPY requirements.txt .
RUN pip wheel --no-cache-dir --wheel-dir /wheels -r requirements.txt

# Runtime stage
FROM python:3.9-slim
COPY --from=builder /wheels /wheels
RUN pip install --no-cache /wheels/*
```

**Result**: Smaller image size, faster deployments

### 2. Model Loading

**Lazy loading**:
```python
model = None

def get_model():
    global model
    if model is None:
        model = load_model()
    return model
```

### 3. Caching

**Add caching middleware**:
```python
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend

@app.on_event("startup")
async def startup():
    redis = aioredis.from_url("redis://localhost")
    FastAPICache.init(RedisBackend(redis), prefix="cache")
```

### 4. Async Processing

**Use background tasks**:
```python
from fastapi import BackgroundTasks

@app.post("/predict/single")
async def predict(file: UploadFile, background_tasks: BackgroundTasks):
    # Process immediately
    result = await process_image(file)
    
    # Log asynchronously
    background_tasks.add_task(log_prediction, result)
    
    return result
```

---

## Security Checklist

- [ ] HTTPS enabled (automatic on Railway/Render)
- [ ] CORS configured for specific origins
- [ ] File size limits enforced
- [ ] Input validation implemented
- [ ] Rate limiting added (production)
- [ ] API authentication (if needed)
- [ ] Secrets stored as environment variables
- [ ] Regular security updates

---

## Cost Estimation

### Free Tiers

| Platform | Free Tier | Limitations |
|----------|-----------|-------------|
| **Railway** | $5 credit/month | ~100 hours uptime |
| **Render** | 750 hours/month | Cold starts after 15min |
| **Fly.io** | 3 shared VMs | Limited bandwidth |
| **Hugging Face** | Unlimited | Community hardware |

### Paid Options

| Platform | Starter Plan | Features |
|----------|--------------|----------|
| **Railway** | $5-20/month | Always-on, more resources |
| **Render** | $7/month | No cold starts |
| **Fly.io** | $1.94/VM/month | More control |

---

## Next Steps

1. ✅ Deploy to chosen platform
2. ✅ Test all API endpoints
3. ✅ Set up monitoring
4. ✅ Configure domain (optional)
5. ✅ Enable analytics
6. ✅ Share your deployment!

---

## Support

- **Railway**: [docs.railway.app](https://docs.railway.app)
- **Render**: [render.com/docs](https://render.com/docs)
- **Fly.io**: [fly.io/docs](https://fly.io/docs)

For project-specific issues, open a GitHub issue.

---

**Last Updated**: November 2025

