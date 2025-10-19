#%% Imports
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

#%% API
app = FastAPI(title = "Diabetes health risk predictor",
              description = "API for predicting health risk based on patient records using RandomForestRegressor",
              version = "v0.2")

class PatientData(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float

model = joblib.load("RFR_regression_model.joblib")

@app.get("/health")
def read_model_details():
    return {"status":"ok", "model_version": "0.2"}

@app.post("/predict")
def predict(data: PatientData):
    input_array = np.array([[data.age, data.sex, data.bmi, data.bp, data.s1, data.s2,
                             data.s3, data.s4, data.s5, data.s6]])

    # Predict progression score
    prediction = round(model.predict(input_array)[0], 2)

    # Compute flag
    HR_threshold = 150
    high_risk_flag = int(prediction >= HR_threshold)
    if high_risk_flag == 1:
        message = "High-risk flag: Patient has high health risk"
    else:
        message = "Patient not flagged for high risk."

    return {
        "predicted_health_risk_score": float(prediction),
        "flag" : message
    }
  