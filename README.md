# Vegetable Image Classifier

A deep learning model for classifying 6 types of Indonesian vegetables and spices:
- Daun Bawang (Scallion)
- Seledri (Celery)
- Jahe (Ginger)
- Lengkuas (Galangal)
- Kunyit (Turmeric)
- Kencur (Lesser Galangal)

## Project Structure

```
.
├── data/
│   ├── daun_bawang/
│   ├── seledri/
│   ├── jahe/
│   ├── lengkuas/
│   ├── kunyit/
│   └── kencur/
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── dataset.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py
│   ├── utils/
│   │   ├── __init__.py
│   │   └── trainer.py
│   ├── __init__.py
│   ├── train.py
│   ├── inference.py
│   └── streamlit_app.py
├── requirements.txt
└── README.md
```

## Training

To train the model:

```bash
python src/train.py
```

The model is trained with the following configuration:
- Batch size: 32
- Learning rate: 0.001
- Number of epochs: 30
- Training/validation split: 80/20

The trained model will be saved in the `checkpoints` directory.

Best model performance achieved:
- Training accuracy: 73.54%
- Validation accuracy: 75.73%

## Inference

The model can be used for inference through either:
1. FastAPI endpoint (`inference.py`)
2. Streamlit web interface (`streamlit_app.py`)

Both methods load the best checkpoint and provide predictions for the 6 vegetable categories.

```
To start the FastAPI server:

```bash
uvicorn src.inference:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### API Endpoints

- `POST /predict`
  - Upload an image file to get the prediction
  - Returns the predicted vegetable class and confidence score

## Model Architecture

The CNN architecture consists of:
- Convolutional layers for feature extraction
- Batch normalization for training stability
- ReLU activation functions
- Max pooling layers for spatial reduction
- Dropouts for regularization
- Fully connected layers for classification

## Streamlit Web Interface

To start the Streamlit web interface:

```bash
streamlit run src/streamlit_app.py
```

The web interface will be available at `http://localhost:8501`.

## Streamlit Community Cloud

 You can access here `https://vegetables-image-classifier.streamlit.app/`
