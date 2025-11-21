"""
Step 2: Inference Script for Breast Histopathology Classification
=================================================================
Predict on single images OR folders of images (like a pathologist!)

Usage:
    # Single image
    python predict.py --image path/to/image.png
    
    # Folder of images (aggregated diagnosis)
    python predict.py --folder path/to/images/
"""

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from collections import Counter


class BreastHistopathologyPredictor:
    """Predictor for breast histopathology images"""
    
    def __init__(self, model_path='models/best_model.pth', device=None):
        """
        Initialize predictor
        
        Args:
            model_path: Path to trained model
            device: 'cuda', 'cpu', or None (auto-detect)
        """
        # Set device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"🖥️  Using device: {self.device}")
        
        # Load model
        print(f"📦 Loading model from: {model_path}")
        self.model = self._load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        print("✅ Model loaded successfully!")
        
        # Class names (matching your notebook: 0=benign, 1=malignant)
        self.class_names = {0: 'benign', 1: 'malignant'}
        
        # Preprocessing transform (matching your notebook exactly)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        # Patch size (matching your training)
        self.patch_size = 224
    
    def _load_model(self, model_path):
        """Load the trained ResNet50 model"""
        # Create architecture (matching your training)
        resnet = models.resnet50(weights=None)
        num_ftrs = resnet.fc.in_features
        resnet.fc = nn.Linear(num_ftrs, 2)  # Binary: benign vs malignant
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        resnet.load_state_dict(checkpoint)
        
        return resnet
    
    def extract_patches(self, image_path):
        """
        Extract 224x224 patches from image (like your training)
        
        Args:
            image_path: Path to image
            
        Returns:
            List of patch tensors
        """
        # Load image
        img = Image.open(image_path).convert('RGB')
        img_array = np.array(img)
        h, w, _ = img_array.shape
        
        patches = []
        patch_coords = []
        
        # Extract non-overlapping patches
        for y in range(0, h - self.patch_size + 1, self.patch_size):
            for x in range(0, w - self.patch_size + 1, self.patch_size):
                patch = img_array[y:y+self.patch_size, x:x+self.patch_size]
                patch_pil = Image.fromarray(patch)
                patch_tensor = self.transform(patch_pil)
                patches.append(patch_tensor)
                patch_coords.append((y, x))
        
        return patches, patch_coords
    
    @torch.no_grad()
    def predict_image(self, image_path, verbose=True):
        """
        Predict diagnosis for a single image
        
        Args:
            image_path: Path to image
            verbose: Print detailed results
            
        Returns:
            Dictionary with prediction results
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔬 Analyzing: {Path(image_path).name}")
            print(f"{'='*60}")
        
        # Extract patches
        patches, coords = self.extract_patches(image_path)
        num_patches = len(patches)
        
        if verbose:
            print(f"📋 Extracted {num_patches} patches (224×224)")
        
        if num_patches == 0:
            raise ValueError("No patches extracted! Image too small?")
        
        # Predict on each patch
        patch_predictions = []
        patch_probabilities = []
        
        for i, patch in enumerate(patches):
            # Add batch dimension and move to device
            patch_batch = patch.unsqueeze(0).to(self.device)
            
            # Get prediction
            output = self.model(patch_batch)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            
            patch_predictions.append(pred)
            patch_probabilities.append(probs.cpu().numpy()[0])
        
        # Calculate statistics
        benign_patches = sum(1 for p in patch_predictions if p == 0)
        malignant_patches = sum(1 for p in patch_predictions if p == 1)
        
        # Average probabilities across all patches
        avg_probs = np.mean(patch_probabilities, axis=0)
        
        # Use AVERAGE PROBABILITY for final prediction (more reliable than MODE)
        final_prediction = np.argmax(avg_probs)
        final_class = self.class_names[final_prediction]
        confidence = float(avg_probs[final_prediction])
        
        result = {
            'image_path': str(image_path),
            'image_name': Path(image_path).name,
            'prediction': final_prediction,
            'prediction_label': final_class,
            'confidence': confidence,
            'num_patches': num_patches,
            'benign_patches': benign_patches,
            'malignant_patches': malignant_patches,
            'avg_prob_benign': float(avg_probs[0]),
            'avg_prob_malignant': float(avg_probs[1]),
            'patch_predictions': patch_predictions
        }
        
        if verbose:
            print(f"\n📊 Patch Analysis:")
            print(f"   Benign patches: {benign_patches}/{num_patches} ({benign_patches/num_patches*100:.1f}%)")
            print(f"   Malignant patches: {malignant_patches}/{num_patches} ({malignant_patches/num_patches*100:.1f}%)")
            
            print(f"\n📈 Average Probabilities:")
            print(f"   Benign: {avg_probs[0]:.4f} ({avg_probs[0]*100:.2f}%)")
            print(f"   Malignant: {avg_probs[1]:.4f} ({avg_probs[1]*100:.2f}%)")
            
            print(f"\n🎯 FINAL DIAGNOSIS (Average Probability):")
            print(f"   Prediction: {final_class.upper()}")
            print(f"   Confidence: {confidence*100:.2f}%")
        
        return result
    
    def predict_folder(self, folder_path, verbose=True):
        """
        Predict diagnosis for folder of images (AGGREGATED - like pathologist!)
        Uses AVERAGE PROBABILITY across all images for final diagnosis
        
        Args:
            folder_path: Path to folder containing images
            verbose: Print detailed results
            
        Returns:
            Dictionary with aggregated results
        """
        folder = Path(folder_path)
        
        # Find all image files
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder.glob(f'*{ext}'))
            image_files.extend(folder.glob(f'*{ext.upper()}'))
        
        image_files = sorted(list(set(image_files)))
        num_images = len(image_files)
        
        if num_images == 0:
            raise ValueError(f"No images found in {folder_path}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🔬 Analyzing Folder: {folder.name}")
            print(f"{'='*60}")
            print(f"📁 Found {num_images} images")
            print(f"{'='*60}")
        
        # Predict each image
        image_results = []
        all_predictions = []
        all_probabilities = []
        
        for i, img_path in enumerate(image_files, 1):
            if verbose:
                print(f"\n[{i}/{num_images}] Processing: {img_path.name}")
            
            result = self.predict_image(img_path, verbose=False)
            image_results.append(result)
            all_predictions.append(result['prediction'])
            all_probabilities.append([result['avg_prob_benign'], result['avg_prob_malignant']])
            
            if verbose:
                pred_label = result['prediction_label'].upper()
                conf = result['confidence'] * 100
                print(f"    → {pred_label} (confidence: {conf:.1f}%)")
        
        # Calculate statistics
        benign_images = sum(1 for p in all_predictions if p == 0)
        malignant_images = sum(1 for p in all_predictions if p == 1)
        
        # Average probabilities across ALL images
        avg_probs = np.mean(all_probabilities, axis=0)
        
        # Use AVERAGE PROBABILITY for final prediction (more reliable than MODE)
        final_prediction = np.argmax(avg_probs)
        final_class = self.class_names[final_prediction]
        confidence = float(avg_probs[final_prediction])
        
        aggregated_result = {
            'folder_path': str(folder_path),
            'folder_name': folder.name,
            'num_images': num_images,
            'final_prediction': final_prediction,
            'final_prediction_label': final_class,
            'confidence': confidence,
            'benign_images': benign_images,
            'malignant_images': malignant_images,
            'avg_prob_benign': float(avg_probs[0]),
            'avg_prob_malignant': float(avg_probs[1]),
            'individual_results': image_results,
            'all_predictions': all_predictions
        }
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"📊 AGGREGATED DIAGNOSIS (across {num_images} images)")
            print(f"{'='*60}")
            print(f"\n📋 Image-Level Results:")
            print(f"   Benign images: {benign_images}/{num_images} ({benign_images/num_images*100:.1f}%)")
            print(f"   Malignant images: {malignant_images}/{num_images} ({malignant_images/num_images*100:.1f}%)")
            
            print(f"\n📈 Average Probabilities (across all images):")
            print(f"   Benign: {avg_probs[0]:.4f} ({avg_probs[0]*100:.2f}%)")
            print(f"   Malignant: {avg_probs[1]:.4f} ({avg_probs[1]*100:.2f}%)")
            
            print(f"\n🎯 FINAL DIAGNOSIS (Average Probability - Recommended):")
            print(f"   Prediction: {final_class.upper()}")
            print(f"   Confidence: {confidence*100:.2f}%")
            print(f"{'='*60}")
        
        return aggregated_result


def main():
    parser = argparse.ArgumentParser(
        description='Breast Histopathology Classification - Inference',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single image
  python predict.py --image test_images/sample.png
  
  # Folder of images (aggregated diagnosis)
  python predict.py --folder test_images/patient_001/
  
  # Use GPU
  python predict.py --image sample.png --device cuda
        """
    )
    
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--image', type=str, help='Path to single image')
    group.add_argument('--folder', type=str, help='Path to folder of images')
    
    parser.add_argument('--model', type=str, default='models/best_model.pth',
                       help='Path to model file (default: models/best_model.pth)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device: cuda or cpu (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    print("\n🔬 Breast Histopathology Classification System")
    print("="*60)
    predictor = BreastHistopathologyPredictor(args.model, args.device)
    
    # Make prediction
    if args.image:
        # Single image mode
        result = predictor.predict_image(args.image)
        
    elif args.folder:
        # Folder mode (aggregated)
        result = predictor.predict_folder(args.folder)
    
    print("\n✅ Analysis complete!")


if __name__ == '__main__':
    main()

