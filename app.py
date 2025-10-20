from fastapi import FastAPI
from pydantic import BaseModel
from RFR_class import RFR_class  # import your model class
import json

app = FastAPI(title="Diabetes Predictor", version="0.2")
MODEL_VERSION = "v0.2"
model_instance = RFR_class()

class DiabetesFeatures(BaseModel):
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

@app.get("/health")
def health():
    try:
        # simple test prediction using zeros
        dummy_input = {feat: 0.0 for feat in model_instance.feature_order}
        _ = model_instance.predict(dummy_input)
        model_status = "ok"
    except Exception:
        model_status = "error"

    return {"status": model_status, "model_version": MODEL_VERSION}

@app.post("/predict")
def predict(features: DiabetesFeatures):
    feature_dict = json.loads(features.model_dump_json())
    prediction = model_instance.predict(feature_dict)
    return {"prediction": prediction}
