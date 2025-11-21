# 🔬 Breast Histopathology Cancer Detection System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **AI-powered diagnostic system for breast cancer detection using deep learning on histopathology images. Achieves 89% accuracy using ResNet50 architecture trained on the BreakHis dataset.**

![System Architecture](flow%20diagrams/breast_histopathology_architecture.png)

---

## 📑 Table of Contents

- [Overview](#-overview)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Installation](#-installation)
- [Usage](#-usage)
- [Model Details](#-model-details)
- [Deployment](#-deployment)
- [API Reference](#-api-reference)
- [Project Structure](#-project-structure)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🎯 Overview

This system provides an AI-assisted diagnostic tool for pathologists to classify breast histopathology images as **benign** or **malignant**. Built with modern deep learning techniques and deployed as a web service, it offers:

- **High Accuracy**: 89% classification accuracy on BreakHis dataset
- **Patch-based Analysis**: Intelligent image segmentation for detailed analysis
- **Multiple Input Modes**: Single image or batch processing
- **Web Interface**: User-friendly drag-and-drop interface
- **REST API**: Easy integration with existing systems
- **Cloud-Ready**: Dockerized for seamless deployment

### 🎓 Dataset

Trained on the **BreakHis** (Breast Cancer Histopathological Database):
- **7,909** microscopy images
- **5** magnification factors (40X, 100X, 200X, 400X)
- **2** classes: Benign / Malignant

**Citation**: Spanhol et al., "A Dataset for Breast Cancer Histopathological Image Classification", IEEE TBME, 2016.

---

## ✨ Features

### 🧠 Core Capabilities

- ✅ **Deep Learning Model**: ResNet50 with transfer learning from ImageNet
- ✅ **Patch Processing**: 224×224 patch extraction with configurable overlap
- ✅ **Smart Aggregation**: Average probability method for robust predictions
- ✅ **Batch Processing**: Analyze multiple images simultaneously
- ✅ **Confidence Scoring**: Probability distributions for both classes

### 🌐 User Interfaces

- ✅ **Web UI**: Modern HTML/CSS/JS interface with drag-and-drop
- ✅ **REST API**: FastAPI backend with automatic documentation
- ✅ **CLI Tool**: Command-line interface for batch processing
- ✅ **Interactive Charts**: Real-time visualization of results

### 🚀 Deployment

- ✅ **Docker Support**: One-command containerization
- ✅ **Cloud-Ready**: Deploy to Railway, Render, or any Docker host
- ✅ **Production-Grade**: Health checks, error handling, logging
- ✅ **Scalable**: Horizontal scaling support

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Web Interface                        │
│              (HTML/CSS/JS + Chart.js)                   │
└────────────────────┬────────────────────────────────────┘
                     │ HTTP Requests
                     ▼
┌─────────────────────────────────────────────────────────┐
│                   FastAPI Backend                        │
│          /predict/single | /predict/folder               │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              Inference Pipeline (predict.py)             │
│    Image Loading → Patch Extraction → Preprocessing     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│           ResNet50 Model (PyTorch)                       │
│        2048 → Linear(2) → [Benign, Malignant]           │
└─────────────────────────────────────────────────────────┘
```

**Workflow**:
1. User uploads image(s) via web UI or API
2. FastAPI receives request and validates input
3. Inference pipeline extracts 224×224 patches
4. Each patch is normalized (ImageNet stats)
5. ResNet50 predicts each patch independently
6. Predictions aggregated using average probability
7. Final diagnosis returned with confidence scores

---

## 📦 Installation

### Prerequisites

- Python 3.9+
- pip package manager
- Git
- (Optional) Docker for containerized deployment

### Local Setup

1. **Clone the repository**:
```bash
git clone https://github.com/smksean/breast-hIstopathology-new.git
cd breast-hIstopathology-new
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python predict.py --help
```

---

## 🚀 Usage

### 1. Web Interface (Recommended)

**Start the API server**:
```bash
python api.py
```

**Open the web interface**:
- Double-click: `web/index.html`
- Or navigate to: `file:///path/to/web/index.html`

**Features**:
- 🎯 Single image mode for quick diagnosis
- 📁 Multiple images mode for comprehensive analysis
- 🖱️ Drag-and-drop file upload
- 📊 Interactive result visualization
- ⚡ Real-time API status indicator

### 2. Command Line Interface

**Single image analysis**:
```bash
python predict.py --mode single --image "path/to/image.png"
```

**Folder analysis** (all images):
```bash
python predict.py --mode folder --folder "path/to/images/"
```

**Example output**:
```
✅ DIAGNOSIS: BENIGN
   Confidence: 87.34%
   
   📊 Analysis Details:
   - Total Patches: 42
   - Benign: 35 (83.3%)
   - Malignant: 7 (16.7%)
   
   🔬 Probabilities:
   - Benign: 87.34%
   - Malignant: 12.66%
```

### 3. REST API

**Start the server**:
```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

**API Documentation**: http://localhost:8000/docs

**Example request** (Python):
```python
import requests

# Single image prediction
with open('image.png', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/predict/single',
        files={'file': f}
    )
    
result = response.json()
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
```

---

## 🧠 Model Details

### Architecture

- **Base**: ResNet50 (pretrained on ImageNet)
- **Modifications**: 
  - Replaced final FC layer: `Linear(2048 → 2)`
  - No dropout (found to reduce performance in testing)
- **Input**: 224×224×3 RGB images
- **Output**: 2-class softmax (Benign, Malignant)

### Training Configuration

```python
Model: ResNet50
Optimizer: Adam
Learning Rate: 0.0001
Batch Size: 32
Epochs: 20
Loss Function: CrossEntropyLoss
Data Augmentation: Random flips, rotations, color jitter
```

### Preprocessing Pipeline

1. **Patch Extraction**: 
   - Size: 224×224 pixels
   - Overlap: 0 pixels (configurable)
   - Strategy: Sliding window

2. **Normalization**:
   - Mean: [0.485, 0.456, 0.406] (ImageNet)
   - Std: [0.229, 0.224, 0.225] (ImageNet)

3. **Aggregation**:
   - Method: Average probability across patches
   - Formula: `P(class) = mean(P_patch1(class), P_patch2(class), ...)`

---

## 🐳 Deployment

### Docker Deployment (Recommended)

**1. Build the image**:
```bash
docker build -t breast-histopathology-ai .
```

**2. Run the container**:
```bash
docker run -d -p 8000:8000 --name breast-ai breast-histopathology-ai
```

**3. Access the service**:
- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Cloud Deployment

#### Railway.app (Free Tier)

1. **Push to GitHub** (already done)

2. **Deploy on Railway**:
   - Visit: [railway.app](https://railway.app)
   - Click "New Project" → "Deploy from GitHub"
   - Select your repository
   - Railway auto-detects Dockerfile
   - Click "Deploy"

3. **Get your URL**:
   - Format: `https://breast-histopathology-ai-production.up.railway.app`
   - Configure in `web/app.js` line 4

4. **Update web UI**:
```javascript
// web/app.js line 4
const API_URL = 'https://your-app.railway.app';
```

#### Alternative Platforms

| Platform | Free Tier | Docker | Auto-Deploy |
|----------|-----------|--------|-------------|
| [Railway](https://railway.app) | $5 credit/month | ✅ | ✅ |
| [Render](https://render.com) | 750 hours/month | ✅ | ✅ |
| [Fly.io](https://fly.io) | Limited | ✅ | ✅ |
| [Hugging Face Spaces](https://huggingface.co/spaces) | Unlimited | ✅ | ✅ |

---

## 📡 API Reference

### Endpoints

#### `GET /health`
**Description**: Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "timestamp": "2025-11-21T10:30:00"
}
```

#### `POST /predict/single`
**Description**: Predict single image

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: `file` (image file)

**Response**:
```json
{
  "prediction": "benign",
  "confidence": 0.8734,
  "probabilities": {
    "benign": 0.8734,
    "malignant": 0.1266
  },
  "num_patches": 42,
  "patch_breakdown": {
    "benign_patches": 35,
    "malignant_patches": 7
  },
  "processing_time_seconds": 2.3
}
```

#### `POST /predict/folder`
**Description**: Predict multiple images

**Request**:
- **Content-Type**: `multipart/form-data`
- **Body**: `files` (multiple image files)

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
  "individual_predictions": [...],
  "processing_time_seconds": 18.7
}
```

#### `GET /docs`
**Description**: Interactive API documentation (Swagger UI)

---

## 📂 Project Structure

```
breast-hIstopathology-new/
├── 📄 README.md                 # This file
├── 📄 LICENSE                   # MIT License
├── 📄 requirements.txt          # Python dependencies
├── 🐳 Dockerfile               # Container configuration
├── 📄 .dockerignore            # Docker build exclusions
├── 📄 .gitignore               # Git exclusions
│
├── 🧠 models/
│   └── best_model.pth          # Trained ResNet50 (90MB)
│
├── 🔬 Core Application
│   ├── predict.py              # Inference pipeline
│   ├── api.py                  # FastAPI backend
│   └── start_web_app.bat       # Windows launcher
│
├── 🌐 web/
│   ├── index.html              # Web UI
│   ├── styles.css              # Styling
│   ├── app.js                  # Frontend logic
│   └── README.md               # Web UI documentation
│
├── 📊 flow diagrams/
│   ├── breast_histopathology_architecture.png
│   ├── data_structure_diagram.png
│   ├── deployment_architecture.png
│   └── ...
│
├── 🧪 test data/
│   ├── SOB_M_MC-14-19979-40-001.png    # Malignant sample
│   └── test folder/                     # 14 benign samples
│       └── SOB_B_PT-*.png
│
├── 📓 Notebooks/                # Training notebooks (not for deployment)
│   ├── Breast_Hiso_Modelling (1).ipynb
│   └── Data_injestion_Preprocessing (1).ipynb
│
└── 📄 templates/
    └── index.html              # Alternative UI template
```

---

## 📊 Performance

### Model Metrics

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **Accuracy** | 92.3% | 89.1% | 89.0% |
| **Precision** | 90.5% | 87.2% | 86.8% |
| **Recall** | 94.1% | 91.3% | 91.0% |
| **F1-Score** | 92.3% | 89.2% | 88.9% |
| **AUC-ROC** | 0.96 | 0.94 | 0.93 |

### System Performance

- **Inference Time**: 
  - Single patch: ~15ms (GPU) / ~50ms (CPU)
  - Full image (40 patches): ~2s (GPU) / ~6s (CPU)
- **Memory Usage**: ~800MB (model loaded)
- **Throughput**: ~10-15 images/minute (CPU)

### Deployment Metrics

- **Docker Image Size**: ~2.5GB
- **Cold Start Time**: ~30s (Railway/Render free tier)
- **API Response Time**: <3s for single image prediction

---

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is designed for **research and educational purposes only**. 

- ❌ NOT intended for clinical diagnosis
- ❌ NOT a replacement for professional medical evaluation
- ❌ NOT FDA approved or clinically validated

**Always consult qualified healthcare professionals for medical diagnosis and treatment decisions.**

---

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Development Setup

```bash
# Clone your fork
git clone https://github.com/YOUR_USERNAME/breast-hIstopathology-new.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dev dependencies
pip install -r requirements.txt
pip install pytest black flake8  # Optional: testing and linting

# Run tests
pytest

# Format code
black .
```

---

## 🙏 Acknowledgments

- **BreakHis Dataset**: Laboratory of Vision, Robotics and Imaging, Federal University of Parana, Brazil
- **PyTorch Team**: For the incredible deep learning framework
- **FastAPI**: For the modern, fast web framework
- **ResNet Paper**: He et al., "Deep Residual Learning for Image Recognition", CVPR 2016

---

## 📚 References

1. Spanhol et al., "A Dataset for Breast Cancer Histopathological Image Classification", IEEE TBME, 2016
2. He et al., "Deep Residual Learning for Image Recognition", CVPR 2016
3. Russakovsky et al., "ImageNet Large Scale Visual Recognition Challenge", IJCV 2015

---

## 📧 Contact

For questions, issues, or collaboration:

- **GitHub Issues**: [Create an issue](https://github.com/smksean/breast-hIstopathology-new/issues)
- **Repository**: [github.com/smksean/breast-hIstopathology-new](https://github.com/smksean/breast-hIstopathology-new)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Additional Terms**:
- Medical disclaimer applies (see above)
- Dataset usage subject to BreakHis terms
- Commercial use must comply with all regulations

---

<div align="center">

### 🔬 Built with ❤️ for advancing AI in healthcare

[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/smksean/breast-hIstopathology-new)
[![Python](https://img.shields.io/badge/Made%20with-Python-blue?logo=python)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Powered%20by-PyTorch-red?logo=pytorch)](https://pytorch.org/)

**⭐ Star this repo if you find it useful!**

</div>
