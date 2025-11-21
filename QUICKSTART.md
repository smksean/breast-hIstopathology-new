# ⚡ Quick Start Guide

Get up and running with Breast Histopathology AI in 5 minutes!

---

## 🚀 For Users

### Option 1: Web Interface (Easiest)

1. **Start the API**:
```bash
python api.py
```

2. **Open Web UI**:
   - Double-click: `web/index.html`
   - Upload images and get instant diagnosis!

### Option 2: Command Line

**Single Image**:
```bash
python predict.py --mode single --image "test data/SOB_M_MC-14-19979-40-001.png"
```

**Multiple Images**:
```bash
python predict.py --mode folder --folder "test data/test folder"
```

---

## 👨‍💻 For Developers

### Local Development

```bash
# Clone
git clone https://github.com/smksean/breast-hIstopathology-new.git
cd breast-hIstopathology-new

# Install
pip install -r requirements.txt

# Run
python api.py
```

### API Testing

```bash
# Test health
curl http://localhost:8000/health

# API docs
open http://localhost:8000/docs
```

---

## 🐳 Docker Deployment

```bash
# Build
docker build -t breast-ai .

# Run
docker run -p 8000:8000 breast-ai
```

---

## ☁️ Cloud Deployment

### Railway (Recommended)

1. Push code to GitHub ✅ (Already done!)
2. Go to [railway.app](https://railway.app)
3. Click "Deploy from GitHub"
4. Select your repo
5. Wait 5 minutes
6. Get your URL!

**Your repo is ready**: https://github.com/smksean/breast-hIstopathology-new

---

## 📚 Documentation

- **README.md**: Full project overview
- **API_DOCUMENTATION.md**: Complete API reference
- **DEPLOYMENT.md**: Detailed deployment guide
- **web/README.md**: Web interface guide

---

## 🆘 Need Help?

**Common Issues**:

❓ **API won't start**
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Kill the process if needed
taskkill /PID <process_id> /F
```

❓ **Model not found**
```bash
# Ensure models folder exists
ls models/best_model.pth
```

❓ **Import errors**
```bash
# Reinstall dependencies
pip install -r requirements.txt
```

---

## ✅ What's Next?

### Already Completed ✨

- ✅ Code cleaned and organized
- ✅ Professional documentation
- ✅ Dockerfile fixed for Railway
- ✅ Pushed to GitHub
- ✅ Ready for deployment

### Next Steps 🎯

1. **Deploy to Railway**:
   - Visit [railway.app](https://railway.app)
   - Deploy from GitHub
   - Get public URL

2. **Update Web UI**:
   - Edit `web/app.js` line 4
   - Change API_URL to your Railway URL
   - Commit and push

3. **Share**:
   - Add deployment URL to README
   - Share with colleagues
   - Get feedback!

---

## 🎉 Your Repository

**GitHub**: https://github.com/smksean/breast-hIstopathology-new

**What's Included**:
- 🧠 Trained ResNet50 model (89% accuracy)
- 🔬 Inference pipeline (CLI & API)
- 🌐 Beautiful web interface
- 🐳 Docker configuration
- 📚 Complete documentation
- 🧪 Test data samples

**Project is 100% ready for deployment!** 🚀

---

## 📊 Test Examples

### Test Malignant Detection

```bash
python predict.py --mode single --image "test data/SOB_M_MC-14-19979-40-001.png"
```

**Expected**: MALIGNANT (100% confidence)

### Test Benign Detection

```bash
python predict.py --mode folder --folder "test data/test folder"
```

**Expected**: BENIGN (63% confidence, 10/14 images benign)

---

**Built with ❤️ | Ready to Deploy 🚀**

