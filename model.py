# model.py (PyTorch version)
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

# ===============================
# Load pretrained emotion model
# ===============================
model = models.mobilenet_v2(weights=None)
model.classifier = nn.Sequential(
    nn.Dropout(0.2),
    nn.Linear(model.last_channel, 128),
    nn.ReLU(),
    nn.Linear(128, 7),
    nn.Softmax(dim=1)
)
model.load_state_dict(torch.load('emotion_model.pth', map_location=torch.device('cpu')))
model.eval()

print("✅ Emotion model loaded successfully.")

# Emotion classes
classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Recommendations
recommendations = {
    'Happy': "Keep smiling! Share your joy with others today.",
    'Sad': "It’s okay to feel down sometimes. Try taking a walk or calling a friend.",
    'Angry': "Take a deep breath. Calmness brings clarity.",
    'Fear': "Courage doesn’t mean no fear, but acting despite it.",
    'Disgust': "Step away from what’s bothering you and refocus your energy.",
    'Surprise': "Wow! Embrace the unexpected moments in life.",
    'Neutral': "A balanced mood is great for productivity."
}

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def predict_emotion(img_path):
    image = Image.open(img_path).convert('RGB')
    img_tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        preds = model(img_tensor)
        predicted_index = preds.argmax(dim=1).item()
        emotion = classes[predicted_index]

    description = recommendations.get(emotion, "Stay positive and mindful!")
    return emotion, description
