# Virtual Diabetes Clinic Triage

This project is a machine learning service designed to simulate the patient triage process in a virtual diabetes clinic. By predicting the short-term disease progression risk for patients, it generates a continuous risk score to help nurses prioritize follow-ups for those who need the most immediate attention.

This repository includes two iterations of model development, a complete API service, and a professional-grade CI/CD (Continuous Integration/Continuous Delivery) pipeline implemented with GitHub Actions.

## Features

### Model Iteration:

* v0.1 (Baseline): Implemented using StandardScaler + LinearRegression.
* v0.2 (Improved): Enhanced with a RandomForestRegressor for better accuracy and introduced a "high-risk" flag.

### API Service:

* Built with FastAPI for high-performance, asynchronous endpoints.
* GET /health: A health check endpoint that returns the service status and model version.
* POST /predict: Accepts patient data and returns the predicted risk score and a risk flag.

### CI/CD Pipeline:

* Continuous Integration (CI): Automatically runs unit tests on push and pull_request to ensure code quality.
* Continuous Delivery (CD): On version tag creation (e.g., v0.2), it automatically builds a Docker image, pushes it to GHCR, runs a smoke test, and creates a GitHub Release.

### Containerization:

* Provides a Dockerfile to package the entire application into a self-contained, portable Docker image.