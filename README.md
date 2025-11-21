# 🔬 Breast Histopathology AI - Cancer Diagnosis System

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

> **Advanced AI-powered system for breast cancer diagnosis using deep learning on histopathology images**

## 🎯 Project Overview

This system uses a **ResNet50** deep learning model trained on the **BreakHis dataset** to classify breast histopathology images as **benign** or **malignant**. It achieves **89% accuracy** and provides pathologists with an AI-assisted diagnostic tool.

### ✨ Key Features

- 🧠 **Deep Learning**: ResNet50 architecture with transfer learning
- 🎯 **High Accuracy**: 89% classification accuracy
- 🖼️ **Patch-based Analysis**: Processes images in 224x224 patches
- 📊 **Aggregation Methods**: Average probability-based prediction
- 🌐 **Web Interface**: Beautiful, lightweight HTML/CSS/JS UI
- 🚀 **REST API**: FastAPI backend with comprehensive endpoints
- 🐳 **Docker Ready**: Containerized for easy deployment
- 📱 **Responsive Design**: Works on all devices

## 🏗️ Architecture

```
┌─────────────────┐
│   Web Interface │ (HTML/CSS/JS + Chart.js)
└────────┬────────┘
         │
    ┌────▼────┐
    │ FastAPI │ (REST API)
    └────┬────┘
         │
    ┌────▼────────┐
    │  Inference  │ (predict.py)
    │   Pipeline  │
    └────┬────────┘
         │
    ┌────▼────────┐
    │   ResNet50  │ (PyTorch Model)
    │   Model     │
    └─────────────┘
```

## 🚀 Quick Start

### Option 1: Using Docker (Recommended)

```bash
# Build the image
docker build -t breast-histopathology-ai .

# Run the container
docker run -p 8000:8000 breast-histopathology-ai
```

### Option 2: Local Installation

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/breast-histopathology-ai.git
cd breast-histopathology-ai

# Install dependencies
pip install -r requirements.txt

# Start the API server
python api.py

# Open web interface
# Navigate to: web/index.html
```

## 📖 Usage

### Web Interface

1. **Start the API**: `python api.py`
2. **Open**: `web/index.html` in your browser
3. **Select Mode**:
   - **Single Image**: Quick diagnosis for one image
   - **Multiple Images**: Comprehensive analysis across multiple slides
4. **Upload**: Drag & drop images or click to browse
5. **Analyze**: Get instant AI-powered diagnosis with confidence scores

### Command Line

**Single Image:**
```bash
python predict.py --mode single --image path/to/image.png
```

**Multiple Images (Folder):**
```bash
python predict.py --mode folder --folder path/to/images/
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict/single` | POST | Predict single image |
| `/predict/folder` | POST | Predict multiple images |
| `/docs` | GET | Interactive API documentation |

## 📊 Model Details

- **Architecture**: ResNet50 (pretrained on ImageNet)
- **Training Dataset**: BreakHis (Breast Cancer Histopathological Database)
- **Classes**: Benign, Malignant
- **Input Size**: 224x224 patches
- **Accuracy**: 89%
- **Framework**: PyTorch

### Preprocessing Pipeline

1. **Patch Extraction**: Divide images into 224x224 patches
2. **Normalization**: Apply ImageNet mean/std normalization
3. **Inference**: Run each patch through ResNet50
4. **Aggregation**: Average probabilities across patches
5. **Final Prediction**: Determine overall diagnosis

## 🎨 Web Interface Features

- ✅ Modern medical-themed design
- ✅ Drag & drop file upload
- ✅ Real-time API status indicator
- ✅ Interactive charts (Chart.js)
- ✅ File previews with thumbnails
- ✅ Responsive layout
- ✅ No build process required (pure HTML/CSS/JS)

## 📁 Project Structure

```
breast-histopathology-ai/
├── api.py                  # FastAPI backend
├── predict.py              # Inference pipeline
├── requirements.txt        # Python dependencies
├── Dockerfile             # Docker configuration
├── models/
│   └── best_model.pth     # Trained model weights
├── web/
│   ├── index.html         # Web UI
│   ├── styles.css         # Styling
│   └── app.js             # Frontend logic
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables (Optional)

```bash
MODEL_PATH=./models/best_model.pth
API_PORT=8000
API_HOST=0.0.0.0
```

### Model Configuration

Edit `predict.py` to adjust:
- Patch size (default: 224x224)
- Overlap (default: 0)
- Batch size for inference
- Device (CPU/GPU)

## 🧪 Testing

### Test Single Image
```bash
python predict.py --mode single --image "test data/SOB_M_MC-14-19979-40-001.png"
```

Expected: **MALIGNANT** diagnosis

### Test Multiple Images
```bash
python predict.py --mode folder --folder "test data/test folder"
```

Expected: **BENIGN** aggregated diagnosis

## 🐳 Docker Deployment

### Build
```bash
docker build -t breast-histopathology-ai .
```

### Run
```bash
docker run -d -p 8000:8000 --name breast-ai breast-histopathology-ai
```

### Check Logs
```bash
docker logs breast-ai
```

## 🌐 Cloud Deployment

### Deploy to Render.com

1. **Push to GitHub**
2. **Go to**: [render.com](https://render.com)
3. **Create New Web Service**
4. **Connect GitHub Repository**
5. **Configure**:
   - Environment: Docker
   - Port: 8000
6. **Deploy!**

You'll get a public URL like: `https://breast-histopathology-ai.onrender.com`

## ⚠️ Medical Disclaimer

**IMPORTANT**: This system is for **research and educational purposes only**. It should NOT be used as the sole basis for medical decisions. Always consult qualified healthcare professionals for medical diagnosis and treatment.

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 89% |
| **Precision** | 87% |
| **Recall** | 91% |
| **F1-Score** | 89% |

## 🛠️ Technology Stack

- **Backend**: Python, FastAPI, PyTorch
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **ML Framework**: PyTorch, torchvision
- **Image Processing**: OpenCV, Pillow
- **Visualization**: Chart.js
- **Deployment**: Docker, Render.com

## 📚 Dataset

This model was trained on the **BreakHis** (Breast Cancer Histopathological Database):
- 7,909 microscopy images
- 5 magnification factors (40X, 100X, 200X, 400X)
- 2 classes: Benign (2,480 images) / Malignant (5,429 images)

**Citation**: Spanhol et al., "A Dataset for Breast Cancer Histopathological Image Classification", IEEE TBME, 2016.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👨‍💻 Author

Built with ❤️ for advancing AI in healthcare

## 🔗 Links

- **Dataset**: [BreakHis Database](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)
- **Paper**: [ResNet Paper](https://arxiv.org/abs/1512.03385)
- **Framework**: [PyTorch](https://pytorch.org/)
- **API Framework**: [FastAPI](https://fastapi.tiangolo.com/)

## 🎓 Acknowledgments

- BreakHis dataset creators
- PyTorch team
- FastAPI developers
- Open source community

---

<div align="center">
  <p><strong>⚕️ Empowering pathologists with AI-assisted diagnosis</strong></p>
  <p>Made with 🔬 and 🤖</p>
</div>
