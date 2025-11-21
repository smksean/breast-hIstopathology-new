"""
FastAPI Backend for Breast Histopathology Classification
=========================================================
RESTful API that wraps the prediction system
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from typing import List
import uvicorn
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import io

# Import our predictor
from predict import BreastHistopathologyPredictor

# Initialize FastAPI app
app = FastAPI(
    title="Breast Histopathology Classification API",
    description="AI-powered classification of breast histopathology images (Benign vs Malignant)",
    version="1.0.0"
)

# Add CORS middleware (allows Streamlit to call the API)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global predictor instance
predictor = None


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global predictor
    
    print("\n" + "="*60)
    print("🚀 Starting Breast Histopathology Classification API")
    print("="*60)
    
    try:
        predictor = BreastHistopathologyPredictor('models/best_model.pth')
        print("✅ Model loaded successfully!")
        print("="*60 + "\n")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("   API will start but predictions will fail.")


@app.get("/api")
async def api_info():
    """API information endpoint"""
    return {
        "message": "Breast Histopathology Classification API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "predict_single": "/predict/single",
            "predict_folder": "/predict/folder",
            "docs": "/docs",
            "api_info": "/api"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_loaded = predictor is not None
    
    return {
        "status": "healthy" if model_loaded else "model_not_loaded",
        "model_loaded": model_loaded,
        "device": str(predictor.device) if model_loaded else "N/A"
    }


@app.post("/predict/single")
async def predict_single_image(file: UploadFile = File(...)):
    """
    Predict diagnosis for a single histopathology image
    
    Args:
        file: Image file (PNG, JPG, etc.)
        
    Returns:
        Prediction results with confidence scores
    """
    # Check if model is loaded
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image."
        )
    
    try:
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmp_file:
            contents = await file.read()
            tmp_file.write(contents)
            tmp_path = tmp_file.name
        
        # Run inference
        result = predictor.predict_image(tmp_path, verbose=False)
        
        # Cleanup
        Path(tmp_path).unlink()
        
        # Format response (convert numpy types to Python native types)
        return {
            "success": True,
            "filename": file.filename,
            "prediction": result['prediction_label'],
            "prediction_idx": int(result['prediction']),
            "confidence": float(result['confidence']),
            "probabilities": {
                "benign": float(result['avg_prob_benign']),
                "malignant": float(result['avg_prob_malignant'])
            },
            "num_patches": int(result['num_patches']),
            "patch_breakdown": {
                "benign_patches": int(result['benign_patches']),
                "malignant_patches": int(result['malignant_patches'])
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.post("/predict/folder")
async def predict_multiple_images(files: List[UploadFile] = File(...)):
    """
    Predict diagnosis for multiple images (folder upload)
    Provides aggregated diagnosis like a pathologist would
    
    Args:
        files: List of image files
        
    Returns:
        Individual predictions + aggregated diagnosis
    """
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please check server logs."
        )
    
    if len(files) == 0:
        raise HTTPException(status_code=400, detail="No files provided")
    
    try:
        # Create temporary directory
        temp_dir = Path(tempfile.mkdtemp())
        temp_files = []
        
        # Save all files
        for file in files:
            if not file.content_type or not file.content_type.startswith('image/'):
                continue
            
            temp_path = temp_dir / file.filename
            contents = await file.read()
            with open(temp_path, 'wb') as f:
                f.write(contents)
            temp_files.append(temp_path)
        
        if len(temp_files) == 0:
            raise HTTPException(status_code=400, detail="No valid image files found")
        
        # Run folder prediction
        result = predictor.predict_folder(temp_dir, verbose=False)
        
        # Cleanup
        shutil.rmtree(temp_dir)
        
        # Format response (convert numpy types to Python native types)
        individual_results = []
        for img_result in result['individual_results']:
            individual_results.append({
                "filename": img_result['image_name'],
                "prediction": img_result['prediction_label'],
                "confidence": float(img_result['confidence']),
                "probabilities": {
                    "benign": float(img_result['avg_prob_benign']),
                    "malignant": float(img_result['avg_prob_malignant'])
                }
            })
        
        return {
            "success": True,
            "num_images": int(result['num_images']),
            "aggregated_diagnosis": {
                "prediction": result['final_prediction_label'],
                "prediction_idx": int(result['final_prediction']),
                "confidence": float(result['confidence']),
                "probabilities": {
                    "benign": float(result['avg_prob_benign']),
                    "malignant": float(result['avg_prob_malignant'])
                }
            },
            "image_breakdown": {
                "benign_images": int(result['benign_images']),
                "malignant_images": int(result['malignant_images'])
            },
            "individual_results": individual_results
        }
        
    except Exception as e:
        # Cleanup on error
        if 'temp_dir' in locals():
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


# Mount static files (web interface) - MUST be last!
# This serves the web UI at the root URL
app.mount("/", StaticFiles(directory="web", html=True), name="static")


def start_server(host: str = "127.0.0.1", port: int = 8000):
    """Start the FastAPI server"""
    print(f"\n🚀 Starting FastAPI server at http://{host}:{port}")
    print(f"📖 API documentation at http://{host}:{port}/docs\n")
    print(f"🌐 Web Interface at http://{host}:{port}/\n")
    
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    start_server()

