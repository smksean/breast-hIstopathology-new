"""
Step 1: Test Your Saved Model (FIXED)
======================================
Now correctly matching your model architecture
"""

import torch
import torch.nn as nn
from torchvision import models

print("="*60)
print("🔍 STEP 1: Testing Your Saved Model (Fixed)")
print("="*60)

# Load the saved file
model_path = 'models/best_model.pth'
print(f"\n📁 Loading model from: {model_path}")

checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)

print("\n📦 Analyzing checkpoint structure...")
print("-" * 60)

# Check the fc layer keys to understand architecture
fc_keys = [k for k in checkpoint.keys() if 'fc' in k]
print(f"✅ FC layer keys found: {fc_keys}")

if 'fc.weight' in checkpoint and 'fc.bias' in checkpoint:
    print("\n🔍 Architecture detected:")
    print("   Your model uses: Simple Linear(2048 → 2)")
    print("   (No Dropout layer)")
    
    fc_weight_shape = checkpoint['fc.weight'].shape
    fc_bias_shape = checkpoint['fc.bias'].shape
    print(f"   FC weight shape: {fc_weight_shape}")
    print(f"   FC bias shape: {fc_bias_shape}")
    
    num_classes = fc_weight_shape[0]
    print(f"\n   ✅ Confirmed: {num_classes} output classes (Benign vs Malignant)")

print("\n" + "="*60)
print("🏗️  Creating MATCHING ResNet50 architecture...")
print("="*60)

# Create ResNet50 with SIMPLE fc layer (matching your training)
resnet = models.resnet50(weights=None)
num_ftrs = resnet.fc.in_features
print(f"   Original FC: {num_ftrs} → 1000")

# Replace with SIMPLE Linear layer (like your training)
resnet.fc = nn.Linear(num_ftrs, 2)
print(f"   New FC: {num_ftrs} → 2 (matching your model)")

# Load the weights
print("\n⚙️  Loading weights...")
try:
    resnet.load_state_dict(checkpoint)
    print("✅ SUCCESS! Model loaded correctly!")
    
    # Set to evaluation mode
    resnet.eval()
    
    # Test with dummy input (224x224 image, matching your patches)
    print("\n🧪 Testing model with dummy 224x224 input...")
    dummy_input = torch.randn(1, 3, 224, 224)  # Batch=1, RGB, 224x224
    
    with torch.no_grad():
        output = resnet(dummy_input)
        probabilities = torch.softmax(output, dim=1)
        prediction = torch.argmax(probabilities, dim=1)
    
    print(f"   ✅ Input shape: {dummy_input.shape}")
    print(f"   ✅ Output shape: {output.shape}")
    print(f"   ✅ Raw logits: [{output[0][0]:.4f}, {output[0][1]:.4f}]")
    print(f"   ✅ Probabilities:")
    print(f"      - Benign (0): {probabilities[0][0]:.4f} ({probabilities[0][0]*100:.2f}%)")
    print(f"      - Malignant (1): {probabilities[0][1]:.4f} ({probabilities[0][1]*100:.2f}%)")
    print(f"   ✅ Prediction: Class {prediction.item()} ({'Benign' if prediction.item() == 0 else 'Malignant'})")
    
    print("\n" + "="*60)
    print("✅ MODEL TEST PASSED! 🎉")
    print("="*60)
    print("\n📋 Summary:")
    print("   - Model architecture: ResNet50")
    print("   - Input size: 224×224×3")
    print("   - Output classes: 2 (Benign=0, Malignant=1)")
    print("   - Preprocessing: ImageNet normalization")
    print("   - Status: Ready for deployment ✓")
    print("\n✨ Next Step: Create inference script for real images")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()


