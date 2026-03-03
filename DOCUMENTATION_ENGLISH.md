# MLOps Diabetes Prediction - Complete Documentation (English)

## Project Overview
This project demonstrates a complete MLOps pipeline for a diabetes prediction application using Machine Learning, FastAPI, Docker, and Kubernetes.

---

## Table of Contents
1. [Project Architecture](#project-architecture)
2. [Step-by-Step Implementation](#step-by-step-implementation)
3. [Code Explanation](#code-explanation)
4. [Deployment Guide](#deployment-guide)
5. [Testing the Application](#testing-the-application)

---

## Project Architecture

```
┌─────────────────┐
│  User Request   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Kubernetes     │
│  LoadBalancer   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FastAPI Pods   │
│  (2 Replicas)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ML Model       │
│  (Random Forest)│
└─────────────────┘
```

---

## Step-by-Step Implementation

### Step 1: Model Training (train.py)

**Purpose**: Train a machine learning model to predict diabetes based on patient data.

**Code**:
```python
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset from a working source (Kaggle/hosted)
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

print("✅ Columns:", df.columns.tolist())

# Prepare data
X = df[["Pregnancies", "Glucose", "BloodPressure", "BMI", "Age"]]
y = df["Outcome"]

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save
joblib.dump(model, "diabetes_model.pkl")
print("✅ Model saved as diabetes_model.pkl")
```

**Explanation**:

1. **Import Libraries**:
   - `pandas`: For data manipulation
   - `sklearn.model_selection.train_test_split`: To split data into training and testing sets
   - `sklearn.ensemble.RandomForestClassifier`: Machine learning algorithm
   - `joblib`: To save the trained model

2. **Load Dataset**:
   - Loads diabetes dataset from GitHub
   - Dataset contains patient information and diabetes outcomes

3. **Prepare Features**:
   - `X`: Input features (Pregnancies, Glucose, BloodPressure, BMI, Age)
   - `y`: Target variable (Outcome - 0 or 1, indicating diabetes)

4. **Split Data**:
   - 80% for training
   - 20% for testing
   - `random_state=42` ensures reproducibility

5. **Train Model**:
   - Uses Random Forest Classifier
   - Random Forest creates multiple decision trees and combines their predictions

6. **Save Model**:
   - Saves trained model as `diabetes_model.pkl`
   - This file will be used by the API

**How to Run**:
```bash
python train.py
```

---

### Step 2: API Development (main.py)

**Purpose**: Create a REST API to serve predictions using the trained model.

**Code**:
```python
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()
model = joblib.load("diabetes_model.pkl")

class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    BMI: float
    Age: int

@app.get("/")
def read_root():
    return {"message": "Diabetes Prediction API is live"}

@app.post("/predict")
def predict(data: DiabetesInput):
    input_data = np.array([[data.Pregnancies, data.Glucose, data.BloodPressure, data.BMI, data.Age]])
    prediction = model.predict(input_data)[0]
    return {"diabetic": bool(prediction)}
```

**Explanation**:

1. **Import Libraries**:
   - `FastAPI`: Modern web framework for building APIs
   - `BaseModel`: For data validation
   - `joblib`: To load the saved model
   - `numpy`: For array operations

2. **Initialize FastAPI**:
   - `app = FastAPI()`: Creates the API application

3. **Load Model**:
   - Loads the previously trained model from `diabetes_model.pkl`

4. **Define Input Schema**:
   - `DiabetesInput` class defines expected input format
   - Ensures data validation (correct types)
   - Fields: Pregnancies (int), Glucose (float), BloodPressure (float), BMI (float), Age (int)

5. **Root Endpoint** (`GET /`):
   - Simple health check endpoint
   - Returns a message confirming API is running

6. **Prediction Endpoint** (`POST /predict`):
   - Accepts patient data in JSON format
   - Converts input to numpy array
   - Uses model to make prediction
   - Returns result as JSON: `{"diabetic": true/false}`

**How to Run Locally**:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Test the API**:
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"Pregnancies": 6, "Glucose": 148, "BloodPressure": 72, "BMI": 33.6, "Age": 50}'
```

---

### Step 3: Dependencies (requirements.txt)

**Purpose**: List all Python packages needed for the project.

**Code**:
```
fastapi
uvicorn[standard]
scikit-learn
pandas
joblib
```

**Explanation**:

1. **fastapi**: Web framework for building the API
2. **uvicorn[standard]**: ASGI server to run FastAPI (with standard features)
3. **scikit-learn**: Machine learning library (includes RandomForestClassifier)
4. **pandas**: Data manipulation and analysis
5. **joblib**: Efficient serialization of Python objects (for saving/loading models)

---

### Step 4: Containerization (Dockerfile)

**Purpose**: Package the application into a Docker container for consistent deployment.

**Code**:
```dockerfile
FROM python:3.10
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Explanation**:

1. **FROM python:3.10**:
   - Base image with Python 3.10 installed
   - Provides the runtime environment

2. **WORKDIR /app**:
   - Sets working directory inside container to `/app`
   - All subsequent commands run from this directory

3. **COPY . /app**:
   - Copies all files from current directory to `/app` in container
   - Includes: main.py, train.py, requirements.txt, diabetes_model.pkl

4. **RUN pip install -r requirements.txt**:
   - Installs all Python dependencies
   - Runs during image build (not at runtime)

5. **CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]**:
   - Command to run when container starts
   - Starts the FastAPI server
   - `--host 0.0.0.0`: Makes server accessible from outside container
   - `--port 8000`: Runs on port 8000

**Build Docker Image**:
```bash
docker build -t anildocker54321/diabetes-prediction-model:latest .
```

**Push to Docker Hub**:
```bash
docker login
docker push anildocker54321/diabetes-prediction-model:latest
```

**Run Container Locally**:
```bash
docker run -p 8000:8000 anildocker54321/diabetes-prediction-model:latest
```

---

### Step 5: Kubernetes Deployment (k8s-deploy.yml)

**Purpose**: Deploy the application to Kubernetes with high availability and load balancing.

**Code**:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: diabetes-api
  labels:
    app: diabetes-api
spec:
  replicas: 2
  selector:
    matchLabels:
      app: diabetes-api
  template:
    metadata:
      labels:
        app: diabetes-api
    spec:
      containers:
      - name: diabetes-api
        image: anildocker54321/diabetes-prediction-model:latest
        ports:
        - containerPort: 8000
        imagePullPolicy: Always
---
apiVersion: v1
kind: Service
metadata:
  name: diabetes-api-service
spec:
  selector:
    app: diabetes-api
  ports:
    - protocol: TCP
      port: 80
      targetPort: 8000
  type: LoadBalancer
```

**Explanation**:

#### Part 1: Deployment

1. **apiVersion: apps/v1**:
   - Kubernetes API version for Deployment resources

2. **kind: Deployment**:
   - Defines this as a Deployment resource
   - Manages pod creation and scaling

3. **metadata**:
   - `name: diabetes-api`: Name of the deployment
   - `labels`: Tags for organizing resources

4. **spec.replicas: 2**:
   - Creates 2 identical pods
   - Provides high availability and load distribution

5. **selector.matchLabels**:
   - Identifies which pods belong to this deployment
   - Matches pods with label `app: diabetes-api`

6. **template**:
   - Defines pod specification
   - Blueprint for creating pods

7. **containers**:
   - `name: diabetes-api`: Container name
   - `image`: Docker image to use
   - `containerPort: 8000`: Port exposed by container
   - `imagePullPolicy: Always`: Always pull latest image from registry

#### Part 2: Service

1. **apiVersion: v1**:
   - Kubernetes API version for Service resources

2. **kind: Service**:
   - Defines this as a Service resource
   - Provides networking and load balancing

3. **metadata.name: diabetes-api-service**:
   - Name of the service

4. **spec.selector**:
   - Routes traffic to pods with label `app: diabetes-api`

5. **ports**:
   - `protocol: TCP`: Uses TCP protocol
   - `port: 80`: External port (what users access)
   - `targetPort: 8000`: Internal port (container port)

6. **type: LoadBalancer**:
   - Creates external load balancer
   - Distributes traffic across pods
   - Provides single entry point

---

## Deployment Guide

### Prerequisites
- Docker installed
- Kubernetes cluster (Minikube/Docker Desktop/Cloud)
- kubectl configured

### Step-by-Step Deployment

#### 1. Train the Model
```bash
python train.py
```
Output: `diabetes_model.pkl` file created

#### 2. Build Docker Image
```bash
docker build -t anildocker54321/diabetes-prediction-model:latest .
```

#### 3. Push to Docker Hub
```bash
docker login
docker push anildocker54321/diabetes-prediction-model:latest
```

#### 4. Start Kubernetes Cluster (Minikube)
```bash
minikube start
```

#### 5. Deploy to Kubernetes
```bash
kubectl apply -f k8s-deploy.yml
```

#### 6. Check Deployment Status
```bash
kubectl get pods
kubectl get services
```

#### 7. Access the Application

**Option A: Minikube Service**
```bash
minikube service diabetes-api-service
```

**Option B: Port Forward**
```bash
kubectl port-forward service/diabetes-api-service 8000:80
```
Access at: http://localhost:8000

**Option C: Minikube Tunnel**
```bash
minikube tunnel
```
Access at: http://localhost

---

## Testing the Application

### 1. Health Check
```bash
curl http://localhost:8000/
```
Expected Response:
```json
{"message": "Diabetes Prediction API is live"}
```

### 2. Make Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 6,
    "Glucose": 148,
    "BloodPressure": 72,
    "BMI": 33.6,
    "Age": 50
  }'
```

Expected Response:
```json
{"diabetic": true}
```

### 3. Test with Different Data
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pregnancies": 1,
    "Glucose": 85,
    "BloodPressure": 66,
    "BMI": 26.6,
    "Age": 31
  }'
```

Expected Response:
```json
{"diabetic": false}
```

---

## Monitoring and Maintenance

### View Logs
```bash
kubectl logs -l app=diabetes-api
```

### Scale Application
```bash
kubectl scale deployment diabetes-api --replicas=5
```

### Update Application
```bash
# Build new image
docker build -t anildocker54321/diabetes-prediction-model:v2 .
docker push anildocker54321/diabetes-prediction-model:v2

# Update deployment
kubectl set image deployment/diabetes-api diabetes-api=anildocker54321/diabetes-prediction-model:v2
```

### Delete Deployment
```bash
kubectl delete -f k8s-deploy.yml
```

---

## Troubleshooting

### Pods Not Starting
```bash
kubectl describe pod <pod-name>
kubectl logs <pod-name>
```

### Image Pull Errors
- Verify image exists in Docker Hub
- Check image name in k8s-deploy.yml
- Ensure Docker Hub credentials are correct

### Service Not Accessible
```bash
kubectl get services
minikube service list
```

---

## Project Structure
```
diabetes-prediction/
├── main.py                 # FastAPI application
├── train.py                # Model training script
├── requirements.txt        # Python dependencies
├── Dockerfile              # Container configuration
├── k8s-deploy.yml          # Kubernetes deployment
├── diabetes_model.pkl      # Trained model (generated)
└── README.md               # Project documentation
```

---

## Key Concepts

### MLOps Pipeline
1. **Model Training**: Create and train ML model
2. **API Development**: Wrap model in REST API
3. **Containerization**: Package in Docker
4. **Orchestration**: Deploy with Kubernetes
5. **Monitoring**: Track performance and logs

### Benefits
- **Scalability**: Easily handle more requests
- **High Availability**: Multiple replicas ensure uptime
- **Portability**: Runs anywhere Docker/Kubernetes runs
- **Maintainability**: Easy updates and rollbacks
- **Monitoring**: Built-in logging and health checks

---

## Conclusion
This project demonstrates a complete MLOps workflow from model training to production deployment using modern DevOps practices and tools.
