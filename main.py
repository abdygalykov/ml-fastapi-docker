import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Iris Classification API",
    description="A simple ML model serving API built with FastAPI and Dockerized for production deployment.",
    version="1.0.0"
)

# Target class labels for the Iris dataset
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Resolve path to the saved model artifact
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model.joblib"))

# Load the trained model artifact
try:
    model = joblib.load(MODEL_PATH)
    print(f"Successfully loaded model artifact from: {MODEL_PATH}")
except Exception as e:
    model = None
    print(f"Error loading model artifact: {e}")

# Define the Pydantic schema for the input features
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", examples=[5.1])
    sepal_width: float = Field(..., description="Sepal width in cm", examples=[3.5])
    petal_length: float = Field(..., description="Petal length in cm", examples=[1.4])
    petal_width: float = Field(..., description="Petal width in cm", examples=[0.2])

    class Config:
        json_schema_extra = {
            "example": {
                "sepal_length": 5.1,
                "sepal_width": 3.5,
                "petal_length": 1.4,
                "petal_width": 0.2
            }
        }

# Endpoint 1: Root endpoint confirming the API is running
@app.get("/")
def read_root():
    return {"message": "ML API is running"}

# Endpoint 2: Prediction endpoint
@app.post("/predict")
def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please ensure model.joblib exists."
        )

    # Convert incoming feature values into numpy array with shape (1, 4)
    data = np.array([[
        features.sepal_length,
        features.sepal_width,
        features.petal_length,
        features.petal_width
    ]])

    # Generate prediction and probabilities
    pred_class_id = int(model.predict(data)[0])
    pred_label = CLASS_NAMES[pred_class_id]

    result = {
        "prediction": pred_label,
        "class_id": pred_class_id
    }

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(data)[0]
        result["probabilities"] = {
            cls: round(float(prob), 4) for cls, prob in zip(CLASS_NAMES, probs)
        }

    return result
