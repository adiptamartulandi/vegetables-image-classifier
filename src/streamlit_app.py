import os
import streamlit as st
import torch
import numpy as np

from PIL import Image
from models.cnn import create_model
from data.dataset import get_transforms


# Page config
st.set_page_config(
    page_title="Vegetable Classifier",
    page_icon="🥬",
    layout="centered"
)

# Title and description
st.title("Vegetable Image Classifier")
st.markdown("""
    Upload an image of any of these vegetables:
    - Green Onion (Daun Bawang)
    - Celery (Seledri)
    - Ginger (Jahe)
    - Galangal (Lengkuas)
    - Turmeric (Kunyit)
    - Lesser Galangal (Kencur)
    
    The model will classify your image and show the prediction result.
""")

# Load model
@st.cache_resource
def load_model():
    device = 'cpu'
    model = create_model(num_classes=6, device=device)
    checkpoint_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'checkpoints', 'best_checkpoint.pth')
    if not os.path.exists(checkpoint_path):
        st.error(f"Model checkpoint not found at {checkpoint_path}. Please ensure the model file is uploaded.")
        st.stop()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, device

# Class mapping
class_names = ['daun_bawang', 'seledri', 'jahe', 'lengkuas', 'kunyit', 'kencur']

# Load the model
model, device = load_model()

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png", "gif", "webp"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Add a prediction button
    if st.button('Predict'):
        # Show spinner during prediction
        with st.spinner('Predicting...'):
            # Preprocess image
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
            
            # Display results
            st.success(f"Prediction: {predicted_class.replace('_', ' ').title()}")
            st.progress(confidence_value)
            st.info(f"Confidence: {confidence_value:.2%}")