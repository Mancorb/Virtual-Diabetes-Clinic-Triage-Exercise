# Virtual Diabetes Clinic Triage

This project is a machine learning service designed to simulate the patient triage process in a virtual diabetes clinic. By predicting the short-term disease progression risk for patients, it generates a continuous risk score to help nurses prioritize follow-ups for those who need the most immediate attention.

This repository includes two iterations of model development, a complete API service, and a professional-grade CI/CD (Continuous Integration/Continuous Delivery) pipeline implemented with GitHub Actions.

## Features

### Model Iteration

* **v0.1 (Baseline):** Implemented using StandardScaler + LinearRegression.
* **v0.2 (Improved):** Enhanced with a RandomForestRegressor for better accuracy and introduced a "high-risk" flag.

### API Service

* Built with FastAPI for high-performance, asynchronous endpoints.
* **GET /health:** A health check endpoint that returns the service status and model version.
* **POST /predict:** Accepts patient data and returns the predicted risk score and a risk flag.

### CI/CD Pipeline

* Continuous Integration (CI): Automatically runs unit tests on push and pull_request to ensure code quality.
* Continuous Delivery (CD): On version tag creation (e.g., v0.2), it automatically builds a Docker image, pushes it to GHCR, runs a smoke test, and creates a GitHub Release.

### Containerization

* Provides a Dockerfile to package the entire application into a self-contained, portable Docker image.

---

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/<your-org>/<your-repo>.git
cd <your-repo>
```
### 2. Install dependencies (local Python environment)

```bash
python -m venv env
source env/bin/activate      # Linux/macOS
env\Scripts\activate         # Windows
pip install -r requirements.txt
```
### 3. Run the FastAPI server locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
### 4. Test Endpoints
```bash
curl http://127.0.0.1:8000/health
```

#### Expected response:
```json
{
  "status": "ok",
  "model_version": "v0.2"
}
```
#### Predict Endpoint:
##### Example Json payload (sample_input.json)

```json
{
  "age": 0.02,
  "sex": -0.044,
  "bmi": 0.06,
  "bp": -0.03,
  "s1": -0.02,
  "s2": 0.03,
  "s3": -0.02,
  "s4": 0.02,
  "s5": 0.02,
  "s6": -0.001
}

```
####Send a request via curl:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
     -H "Content-Type: application/json" \
     -d @sample_input.json
```
#### Expected response:
```json
{
  "prediction": 123.456789
}

```
### Run unit tests:
```bash
pytest -v
```
### 6. Docker Usage
Build docker Image
```bash
docker build -t ghcr.io/<org>/<repo>:v0.2 .
```
Run Docker Container
```bash
docker run -p 8000:8000 ghcr.io/<org>/<repo>:v0.2
```
Test Endpoints in container
* Health: ```curl http://localhost:8000/health```
* Predict: ```curl -X POST -H "Content-Type: application/json" -d @sample_input.json http://localhost:8000/predict```

