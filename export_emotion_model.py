# export_emotion_model.py (PyTorch version)

import torch
import torch.nn as nn
import torchvision.models as models

# ===========================
# Step 1: Load Pretrained Model
# ===========================
base_model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)

# Freeze base model layers
for param in base_model.features.parameters():
    param.requires_grad = False

# ===========================
# Step 2: Add Custom Classifier
# ===========================
num_classes = 7  # same as before
base_model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(base_model.last_channel, 128),
    nn.ReLU(),
    nn.Linear(128, num_classes),
    nn.Softmax(dim=1)
)

# ===========================
# Step 3: Save Model
# ===========================
torch.save(base_model.state_dict(), 'emotion_model.pth')
print("✅ Model exported successfully as emotion_model.pth")
