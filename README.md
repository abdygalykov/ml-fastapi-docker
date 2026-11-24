# Practical Task: Deploy ML Model with FastAPI and Docker

This project demonstrates the end-to-end workflow of training a Machine Learning classification model, exposing it via a RESTful API using **FastAPI**, and packaging it into a reproducible **Docker** container.

---

## 📁 Project Structure

```text
ml-fastapi-docker/
│
├── main.py                     # FastAPI application serving ML predictions
├── train.py                    # Script to train and serialize the ML model
├── model.joblib                # Serialized trained model artifact
├── requirements.txt            # Python dependencies
├── Dockerfile                  # Container build instructions
├── README.md                   # Project documentation and run guide
├── ml_fastapi_docker.ipynb     # Interactive Jupyter Notebook walkthrough
└── .gitignore                  # Git ignore file for venv and caches
```

### File Roles

* **`train.py`**: Loads the classical Iris dataset, trains a `RandomForestClassifier`, evaluates its test accuracy, and exports the serialized model artifact to `model.joblib`.
* **`main.py`**: Implements the FastAPI web server with Pydantic request validation and two primary endpoints (`GET /` and `POST /predict`).
* **`model.joblib`**: Serialized binary representation of the trained model artifact loaded into memory during API startup.
* **`requirements.txt`**: Specifies exact Python library versions required to run the application.
* **`Dockerfile`**: Defines the containerization steps: base image, dependency installation, code copying, port exposure, and server launch.
* **`README.md`**: Complete documentation covering architecture, setup, and verification.
* **`ml_fastapi_docker.ipynb`**: Complete Jupyter notebook walking through every step interactively with outputs.

---

## 🚀 Step-by-Step Instructions

### Step 1: Train the Machine Learning Model

Activate your environment and run the training script:

```bash
python train.py
```

**Expected Output:**
```text
1. Loading dataset (Iris)...
Dataset shape: (150, 4), Classes: ['setosa', 'versicolor', 'virginica']
2. Training RandomForestClassifier...
Test Accuracy: 90.00%
3. Model successfully saved to 'model.joblib'
```

---

### Step 2 & 3: Run and Test FastAPI Locally

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Start the API locally using Uvicorn:**
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```

3. **Verify Endpoints:**

   * **Root Endpoint (`GET /`):**
     ```bash
     curl -X GET http://127.0.0.1:8000/
     ```
     Response:
     ```json
     {"message": "ML API is running"}
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

   * **Interactive Swagger UI:**
     Open your browser and navigate to:
     [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### Step 4 & 5: Build the Docker Image

1. **Build the container image:**
   ```bash
   docker build -t ml-fastapi-app .
   ```

2. **Verify the built image:**
   ```bash
   docker images | grep ml-fastapi-app
   ```

---

### Step 6: Run the Docker Container

1. **Run the container in detached mode with port 8000 exposed:**
   ```bash
   docker run -d -p 8000:8000 --name ml-api-container ml-fastapi-app
   ```

2. **Verify container status:**
   ```bash
   docker ps
   docker logs ml-api-container
   ```

3. **Test the containerized API:**
   ```bash
   # Check root endpoint
   curl http://localhost:8000/

   # Send test prediction request
   curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"sepal_length": 6.7, "sepal_width": 3.0, "petal_length": 5.2, "petal_width": 2.3}'
   ```
   Expected response:
   ```json
   {
     "prediction": "virginica",
     "class_id": 2,
     "probabilities": {
       "setosa": 0.0,
       "versicolor": 0.05,
       "virginica": 0.95
     }
   }
   ```

4. **Stop and clean up the container:**
   ```bash
   docker stop ml-api-container
   docker rm ml-api-container
   ```

---

## 📤 Git Submission Checklist

To upload your Git Link for this project:

```bash
git init
git add .
git commit -m "Complete Practical Task: Deploy ML Model with FastAPI and Docker"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo-name>.git
git push -u origin main
```
