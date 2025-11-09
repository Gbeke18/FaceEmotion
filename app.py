from flask import Flask, render_template, request, jsonify
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import uuid

# Initialize Flask app
app = Flask(__name__)

# Define upload folder
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ===============================
# Load trained emotion model
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

# ===============================
# Emotion labels and descriptions
# ===============================
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
emotion_descriptions = {
    'Angry': 'You seem upset or frustrated.',
    'Disgust': 'That face shows dislike or displeasure.',
    'Fear': 'Looks like you are frightened or anxious.',
    'Happy': 'A bright smile! You look happy.',
    'Neutral': 'A calm and neutral expression.',
    'Sad': 'You look sad or disappointed.',
    'Surprise': 'You look surprised or shocked.'
}

# Preprocessing for PyTorch
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        file = request.files['file']
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400

        filename = f"{uuid.uuid4().hex}.jpg"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        image = Image.open(filepath).convert('RGB')
        img_tensor = transform(image).unsqueeze(0)

        with torch.no_grad():
            preds = model(img_tensor)
            emotion_idx = preds.argmax(dim=1).item()
            emotion = emotion_labels[emotion_idx]
            description = emotion_descriptions.get(emotion, "")

        return jsonify({'emotion': emotion, 'description': description})

    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
