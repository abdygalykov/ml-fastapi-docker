import os
import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Initialize FastAPI application
app = FastAPI(
    title="Iris Classification API",
    description="A production-ready ML model serving API built with FastAPI, integrated with MLflow, and containerized with Docker.",
    version="2.0.0"
)

# Enable CORS for frontend clients (e.g., Streamlit on port 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Target class labels for the Iris dataset
CLASS_NAMES = ["setosa", "versicolor", "virginica"]

# Model loading logic: MLflow registry support with local fallback
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.getenv("MODEL_PATH", os.path.join(BASE_DIR, "model.joblib"))
MLFLOW_MODEL_URI = os.getenv("MLFLOW_MODEL_URI", None)

model = None
model_source = "None"

# Attempt 1: Load from MLflow if URI is explicitly specified
if MLFLOW_MODEL_URI:
    try:
        import mlflow.sklearn
        model = mlflow.sklearn.load_model(MLFLOW_MODEL_URI)
        model_source = f"MLflow URI ({MLFLOW_MODEL_URI})"
        print(f"Successfully loaded model from MLflow: {MLFLOW_MODEL_URI}")
    except Exception as mlflow_err:
        print(f"Notice: Could not load from MLflow URI ({mlflow_err}), falling back to local file.")

# Attempt 2: Fallback to local joblib artifact
if model is None:
    try:
        model = joblib.load(MODEL_PATH)
        model_source = f"Local File ({os.path.basename(MODEL_PATH)})"
        print(f"Successfully loaded model artifact from: {MODEL_PATH}")
    except Exception as file_err:
        print(f"Error loading local model artifact: {file_err}")
        model_source = "Unavailable"


# Pydantic schema for input feature validation
class IrisFeatures(BaseModel):
    sepal_length: float = Field(..., description="Sepal length in cm", ge=0.0, le=15.0, examples=[5.1])
    sepal_width: float = Field(..., description="Sepal width in cm", ge=0.0, le=15.0, examples=[3.5])
    petal_length: float = Field(..., description="Petal length in cm", ge=0.0, le=15.0, examples=[1.4])
    petal_width: float = Field(..., description="Petal width in cm", ge=0.0, le=15.0, examples=[0.2])

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
    return {
        "message": "ML API is running",
        "status": "healthy",
        "version": "2.0.0"
    }


# Endpoint 2: Model information and metadata endpoint
@app.get("/model-info")
def model_info():
    return {
        "model_name": "IrisClassifier",
        "algorithm": "RandomForestClassifier",
        "classes": CLASS_NAMES,
        "features": ["sepal_length", "sepal_width", "petal_length", "petal_width"],
        "model_source": model_source,
        "is_loaded": model is not None
    }


# Endpoint 3: Prediction endpoint
@app.post("/predict")
def predict(features: IrisFeatures):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail="Model is not loaded. Please ensure model.joblib exists or train.py has been executed."
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