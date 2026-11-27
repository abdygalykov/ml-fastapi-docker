# SIS - 3: Complete Machine Learning System Design
## Continuation of Practical Task 6: FastAPI, Streamlit, MLflow & Docker

This project extends **Practical Task 6** into a full-lifecycle, production-ready Machine Learning system design. It connects model training, experiment tracking, model registry, an inference REST API, an interactive web frontend, and containerized multi-service orchestration.

---

## 🏗️ System Architecture

```text
                               +-------------------------------------+
                               |           User Browser              |
                               +------------------+------------------+
                                                  |
                                   HTTP (Port 8501) |
                                                  v
                               +-------------------------------------+
                               |     Streamlit Web Frontend          |
                               |     (frontend/app.py)               |
                               |  - Interactive feature sliders      |
                               |  - Real-time confidence bar chart   |
                               |  - Quick preset buttons             |
                               +------------------+------------------+
                                                  |
                                   HTTP (Port 8000) |
                                                  v
                               +-------------------------------------+
                               |       FastAPI REST Service          |
                               |       (main.py)                     |
                               |  - GET / (Health check)             |
                               |  - GET /model-info (Metadata)       |
                               |  - POST /predict (Inference engine) |
                               +---------+-----------------+---------+
                                         |                 |
                        Local fallback   |                 | Registry URI
                                         v                 v
                      +-----------------------+   +-----------------------+
                      |      model.joblib     |   |  MLflow Model Registry|
                      |  - Serialized model   |   |  - Model: IrisClassifier
                      |  - Local artifact     |   |  - Versioned runs & tags
                      +-----------------------+   +-----------+-----------+
                                                              |
                                             HTTP (Port 5000) |
                                                              v
                                                  +-----------------------+
                                                  |     MLflow Web UI     |
                                                  |  - Experiment Tracking|
                                                  |  - Metrics & Artifacts|
                                                  +-----------------------+
```

---

## 📁 Project Structure

```text
ml-fastapi-docker/
│
├── main.py                     # FastAPI application serving ML predictions & metadata
├── train.py                    # Training script with MLflow Tracking & Model Registry
├── model.joblib                # Serialized trained model artifact
├── confusion_matrix.png        # Generated evaluation plot artifact logged to MLflow
├── classification_report.txt   # Generated evaluation metrics text artifact
│
├── frontend/
│   ├── app.py                  # Streamlit interactive frontend application
│   └── requirements.txt        # Dedicated dependencies for the frontend container
│
├── requirements.txt            # Project dependencies (FastAPI, MLflow, Scikit-learn, etc.)
├── Dockerfile                  # Dockerfile for FastAPI backend & MLflow UI
├── Dockerfile.frontend         # Dockerfile for Streamlit frontend
├── docker-compose.yml          # Multi-container orchestration (FastAPI + Streamlit + MLflow)
│
├── sis3_mlflow_streamlit.ipynb # Complete interactive Jupyter Notebook for SIS-3
├── ml_fastapi_docker.ipynb     # Previous Practical Task 6 walkthrough notebook
├── .dockerignore               # Files excluded from Docker builds
├── .gitignore                  # Git ignore rules (venv, caches, etc.)
└── README.md                   # Comprehensive system documentation
```

### File Responsibilities
* **`train.py`**: Loads the Iris dataset, trains a `RandomForestClassifier`, evaluates test metrics (accuracy, F1-score, precision, recall), logs parameters, metrics, and artifacts to MLflow, and registers the model in the **MLflow Model Registry** under `IrisClassifier`.
* **`main.py`**: Exposes FastAPI endpoints (`GET /`, `GET /model-info`, `POST /predict`) with Pydantic request validation and CORS support for the Streamlit frontend.
* **`frontend/app.py`**: Streamlit web interface with real-time prediction sliders, sample presets, confidence probability visualizations, batch testing, and architecture documentation.
* **`docker-compose.yml`**: Defines a multi-container network hosting the API (port 8000), Frontend (port 8501), and MLflow UI (port 5000).

---

## ⚙️ Prerequisites & Installation

Create and activate a virtual environment, then install dependencies:

```bash
# 1. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 2. Install all dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Step-by-Step Instructions

### Step 1: Run MLflow Experiment Tracking & Model Registration

Execute the training script:
```bash
python train.py
```

**What this performs:**
1. **Experiment Tracking:**
   * Creates/sets experiment: `iris_classification_experiment`.
   * Logs hyperparameters: `n_estimators=100`, `max_depth=5`, `criterion=gini`, `random_state=42`, `test_size=0.2`.
   * Logs evaluation metrics: `accuracy`, `f1_score_macro`, `f1_score_weighted`, `precision_macro`, `recall_macro`.
   * Logs artifacts: `model.joblib`, `confusion_matrix.png`, and `classification_report.txt`.
2. **Model Registry:**
   * Registers the model in the MLflow Model Registry under name: **`IrisClassifier`**.
   * Assigns model version (e.g., Version 1).
   * Sets version tags (`stage: Production`) and metadata description.

**Expected Terminal Output:**
```text
============================================================
Step 1: Loading Iris Dataset...
Dataset Shape: (150, 4) samples, 4 features
Target Classes: ['setosa', 'versicolor', 'virginica']

============================================================
Step 2: Training RandomForestClassifier...

============================================================
Step 3: Evaluating Model Performance...
Test Accuracy: 100.00%
F1-Score (Macro): 1.0000
Precision (Macro): 1.0000
Recall (Macro): 1.0000

============================================================
Step 4: MLflow Experiment Tracking & Model Registration...
MLflow Tracking URI: sqlite:///mlflow.db
Active Experiment: 'iris_classification_experiment'
Started MLflow Run ID: 6e9b4...
Logging parameters to MLflow...
Logging evaluation metrics to MLflow...
Logging artifacts to MLflow...
Successfully registered model 'IrisClassifier' Version: 1
============================================================
MLflow Tracking & Registration Completed Successfully!
```

---

### Step 2: Launch and Inspect MLflow Web UI

Start the MLflow tracking server locally:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000
```

* Open your browser at: **[http://localhost:5000](http://localhost:5000)**
* **Experiments Tab:** View logged parameters (`n_estimators`, `max_depth`), evaluation metrics (`accuracy`, `f1_score`), and artifacts (`model.joblib`, `confusion_matrix.png`).
* **Models Tab (Model Registry):** Inspect the registered model `IrisClassifier`, its active versions, and production tags.

---

### Step 3: Run and Test the FastAPI Backend

1. **Start the API locally using Uvicorn:**
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

2. **Verify Endpoints:**

   * **Health Check (`GET /`):**
     ```bash
     curl -X GET http://127.0.0.1:8000/
     ```
     Response:
     ```json
     {"message": "ML API is running", "status": "healthy", "version": "2.0.0"}
     ```

   * **Model Metadata (`GET /model-info`):**
     ```bash
     curl -X GET http://127.0.0.1:8000/model-info
     ```

   * **Prediction Endpoint (`POST /predict`):**
     ```bash
     curl -X POST http://127.0.0.1:8000/predict \
       -H "Content-Type: application/json" \
       -d '{
         "sepal_length": 5.1,
         "sepal_width": 3.5,
         "petal_length": 1.4,
         "petal_width": 0.2
       }'
     ```
     Response:
     ```json
     {
       "prediction": "setosa",
       "class_id": 0,
       "probabilities": {
         "setosa": 1.0,
         "versicolor": 0.0,
         "virginica": 0.0
       }
     }
     ```

   * **Swagger Interactive Docs:**
     Open: **[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)**

---

### Step 4: Run the Streamlit Interactive Frontend

Open a new terminal and run:

```bash
streamlit run frontend/app.py --server.port 8501
```

* Open your browser at: **[http://localhost:8501](http://localhost:8501)**
* **Features:**
  1. **Interactive Controls:** Move sliders for Sepal & Petal dimensions.
  2. **Preset Buttons:** Click "🌸 Iris Setosa", "🌿 Iris Versicolor", or "🌺 Iris Virginica" in the sidebar to load benchmark values.
  3. **Visual Output:** Instant display of predicted species with confidence probability distribution bar chart.
  4. **Batch Testing Tab:** Evaluate multiple test cases concurrently.
  5. **Architecture Tab:** Review the system pipeline design.

---

### Step 5: Multi-Container Deployment with Docker Compose

To build and run all 3 services concurrently in isolated containers:

```bash
docker compose up --build
```

**Services Running:**
| Service | Container Name | Port | Description |
| :--- | :--- | :--- | :--- |
| **Streamlit UI** | `ml-streamlit-frontend` | `http://localhost:8501` | Interactive User Frontend |
| **FastAPI** | `ml-fastapi-backend` | `http://localhost:8000` | REST API Inference Engine |
| **MLflow UI** | `ml-mlflow-ui` | `http://localhost:5000` | Experiment Tracking & Registry UI |

**Verify Running Containers:**
```bash
docker compose ps
```

**Stop and Clean Up Containers:**
```bash
docker compose down
```

---

## 📤 Git Submission Guide

To update your GitHub repository:

```bash
# 1. Stage all files
git add .

# 2. Commit changes
git commit -m "Complete SIS-3: Add Streamlit frontend, MLflow experiment tracking & model registry, and Docker Compose"

# 3. Push to GitHub
git push origin main
```