#Imports
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import joblib
import numpy as np

#API
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

try:
    model = joblib.load("RFR_regression_model.joblib")
except FileNotFoundError:
    raise RuntimeError("Model file missing: please ensure RFR_regression_model.joblib is in /Virtual-Diabetes-Clinic-Triage-Exercise")


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Return clear JSON when invalid input is sent"""
    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid input format",
            "message": "Please ensure all 10 input fields are filled in and that input is numeric.",
            "details": exc.errors()
        },
    )

@app.get("/health")
def read_model_details():
    return {"status":"ok", "model_version": "0.2"}

@app.post("/predict")
def predict(data: PatientData):
    try:
        input_array = np.array([[data.age, data.sex, data.bmi, data.bp, data.s1, data.s2,
                                 data.s3, data.s4, data.s5, data.s6]])
        prediction = round(model.predict(input_array)[0], 2)

        HR_threshold = 150
        high_risk_flag = int(prediction >= HR_threshold)
        message = (
            "High-risk flag: Patient has high health risk"
            if high_risk_flag == 1
            else "Patient not flagged for high risk."
        )

        return {
            "predicted_health_risk_score": float(prediction),
            "flag": message
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Bad input: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")
  