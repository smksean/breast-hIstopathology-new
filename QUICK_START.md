# ⚡ Quick Start Guide

Get your Breast Histopathology Classification system running in 5 minutes!

## 🎯 Prerequisites

- Python 3.10+
- Your trained model checkpoint (`best_model.pth`)
- 4GB RAM minimum

## 🚀 3-Step Setup

### Step 1: Install Dependencies (1 minute)

```bash
# Clone repository
git clone https://github.com/yourusername/breast-histopathology.git
cd breast-histopathology

# Install requirements
pip install -r requirements.txt
```

### Step 2: Export Model (1 minute)

```bash
# Export your trained model
python model_export.py /path/to/your/best_model.pth
```

**Example**:
```bash
# If your model is in Google Drive
python model_export.py "/content/drive/MyDrive/BreakHis/checkpoints/best_model.pth"

# Local path
python model_export.py "C:\Users\user\Desktop\models\best_model.pth"
```

### Step 3: Start the System (1 minute)

```bash
# Start API server
python app.py

# In another terminal, serve frontend
cd templates
python -m http.server 8080
```

## 🎉 You're Ready!

- **API**: http://localhost:8000
- **Web Interface**: http://localhost:8080
- **API Docs**: http://localhost:8000/docs

## 📝 Test It

### Option 1: Web Interface

1. Open http://localhost:8080
2. Drag and drop an image
3. Click "Analyze Image(s)"
4. View results!

### Option 2: Command Line

```bash
python inference_pipeline.py \
    deployment/models/breast_histopathology_resnet50.pth \
    deployment/models/model_metadata.json \
    /path/to/test/image.png
```

### Option 3: API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -F "file=@/path/to/image.png"
```

### Option 4: Python Code

```python
from inference_pipeline import InferencePipeline

pipeline = InferencePipeline(
    model_path='deployment/models/breast_histopathology_resnet50.pth',
    metadata_path='deployment/models/model_metadata.json'
)

result = pipeline.predict_image('path/to/image.png')
print(f"Prediction: {result['predicted_class']}")
print(f"Confidence: {result['confidence']:.2%}")
```

## 🐳 Quick Start with Docker

Even faster with Docker:

```bash
# Build and run
docker-compose up -d

# Access at http://localhost:8000
```

## ❓ Having Issues?

### Issue: "Model file not found"
```bash
# Make sure you ran the export script
python model_export.py /path/to/best_model.pth

# Check if files exist
ls deployment/models/
```

### Issue: "Port already in use"
```bash
# Use different port
python app.py --port 8001

# Or kill existing process
# Windows:
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -ti:8000 | xargs kill -9
```

### Issue: "CUDA out of memory"
```bash
# Use CPU instead
# In app.py, set device='cpu' in InferencePipeline initialization
```

## 📚 Next Steps

- Read the [full README](README.md) for details
- Check [DEPLOYMENT_GUIDE](DEPLOYMENT_GUIDE.md) for production setup
- Explore the [API documentation](http://localhost:8000/docs)

## 🎓 Understanding the System

### What happens when you upload an image?

1. **Patch Extraction**: Image is split into 224×224 patches
2. **Preprocessing**: Each patch is normalized
3. **Model Inference**: ResNet50 predicts each patch
4. **Aggregation**: Majority voting determines final result
5. **Results**: You get prediction + confidence + probabilities

### File Structure After Setup

```
breast-histopathology/
├── app.py                              # ✅ API server
├── inference_pipeline.py               # ✅ Prediction logic
├── model_export.py                     # ✅ Model exporter
├── requirements.txt                    # ✅ Dependencies
├── templates/
│   └── index.html                      # ✅ Web interface
└── deployment/
    └── models/                         # ✅ Your exported model
        ├── breast_histopathology_resnet50.pth
        ├── model_metadata.json
        └── breast_histopathology_resnet50_traced.pt
```

## 💡 Pro Tips

1. **GPU Acceleration**: If you have CUDA-compatible GPU, inference will be automatic and faster
2. **Batch Processing**: Use "Batch Mode" in web interface for multiple images
3. **Slide-Level Analysis**: Upload multiple images from same slide for clinical diagnosis
4. **API Integration**: Use the REST API to integrate with your own applications

## 🎬 Video Tutorial

[Watch Quick Start Tutorial](https://youtu.be/your-video-link) (Coming soon!)

## ✅ Checklist

- [ ] Installed dependencies
- [ ] Exported model
- [ ] Started API server
- [ ] Tested with sample image
- [ ] Explored web interface
- [ ] Read API documentation

## 🆘 Get Help

- **Documentation**: [README.md](README.md)
- **Issues**: [GitHub Issues](https://github.com/yourusername/breast-histopathology/issues)
- **Email**: your.email@example.com

---

**Time to First Prediction**: ~5 minutes ⏱️

Happy Predicting! 🔬🎉

