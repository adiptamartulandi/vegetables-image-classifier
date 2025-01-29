import io
import torch
import numpy as np
import base64

from typing import Dict
from PIL import Image, ImageDraw, ImageFont
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse

from src.models.cnn import create_model
from src.data.dataset import get_transforms


app = FastAPI(title="Vegetable Classifier API")

# Load model
device = 'cpu'
model = create_model(num_classes=2, device=device)
checkpoint = torch.load('checkpoints/best_checkpoint.pth', map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Class mapping
class_names = ['daun_bawang', 'seledri']

def draw_prediction(image: Image.Image, class_name: str, confidence: float) -> Image.Image:
    # Create a copy of the image
    result_image = image.copy()
    draw = ImageDraw.Draw(result_image)
    
    # Set up text parameters
    text = f"{class_name} ({confidence:.1%})"
    
    # Calculate text position (bottom-left corner)
    margin = 10
    text_position = (margin, image.height - 30 - margin)
    
    # Create semi-transparent background for text
    text_bbox = draw.textbbox(text_position, text)
    draw.rectangle(
        [text_bbox[0]-5, text_bbox[1]-5, text_bbox[2]+5, text_bbox[3]+5],
        fill=(0, 0, 0, 128)
    )
    
    # Draw text
    draw.text(text_position, text, fill=(255, 255, 255))
    
    return result_image

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
    
    # Draw prediction on image
    result_image = draw_prediction(image, predicted_class, confidence_value)
    
    # Convert result image to base64
    buffered = io.BytesIO()
    result_image.save(buffered, format="JPEG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    
    return {
        "class": predicted_class,
        "confidence": f"{confidence_value:.2%}",
        "image": f"data:image/jpeg;base64,{img_str}"
    }

@app.get("/")
async def root() -> Dict[str, str]:
    return {"message": "Welcome to the Vegetable Classifier API"}