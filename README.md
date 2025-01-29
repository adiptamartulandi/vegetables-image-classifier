# Vegetable Image Classifier

A PyTorch-based CNN classifier for distinguishing between two types of vegetables: Daun Bawang (Green Onion) and Seledri (Celery).

## Project Structure

```
.
├── data/
│   ├── daun_bawang/
│   └── seledri/
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
│   └── inference.py
├── requirements.txt
└── README.md
```
## Dataset

The dataset contains two classes of vegetable images that acquire by using SerpApi [here](https://github.com/adiptamartulandi/google-image-scraper):
- Daun Bawang (Green Onion)
- Seledri (Celery)

Data augmentation techniques are applied during training to improve model generalization.

## Setup

1. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Training

To train the model:

```bash
python src/train.py
```

The trained model will be saved in the `checkpoints` directory.

Best Models in epoch 7 with:
- train accuracy 77.30%
- val accuracy 80.56%

## Inference API

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
