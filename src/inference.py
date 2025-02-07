import io
import torch
import numpy as np
import base64

from typing import Dict
from PIL import Image
from fastapi import FastAPI, UploadFile, File

from src.models.cnn import create_model
from src.data.dataset import get_transforms


app = FastAPI(title="Vegetable Classifier API")

# Load model
device = 'cpu'
model = create_model(num_classes=6, device=device)
checkpoint = torch.load('checkpoints/best_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Class mapping
class_names = ['daun_bawang', 'seledri', 'jahe', 'lengkuas', 'kunyit', 'kencur']

@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> Dict[str, str]:
    # Read and preprocess image
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Apply transforms
    transform = get_transforms(train=False)
    image_tensor = transform(image=np.array(image))['image']
    image_tensor = image_tensor.unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    
    # Get prediction results
    predicted_class = class_names[predicted.item()]
    confidence_value = confidence.item()
    
    return {
        "class": predicted_class,
        "confidence": f"{confidence_value:.2%}"
    }

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Welcome to the Vegetable Classifier API"}