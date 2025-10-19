from fastapi.testclient import TestClient
from API import app

client = TestClient(app)

def test_health_check():
    """
    Tests if the /health endpoint returns a 200 OK status and the correct JSON response.
    """
    response = client.get("/health")
    
    assert response.status_code == 200

    assert response.json() == {"status": "ok", "model_version": "0.2"}


def test_predict_success():
    """
    Tests the /predict endpoint with a valid patient data payload.
    """
    valid_patient_data = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
        "s5": 0.02, "s6": -0.001
    }
    response = client.post("/predict", json=valid_patient_data)
    
    assert response.status_code == 200
    
    response_data = response.json()
    assert "predicted_health_risk_score" in response_data

    assert "flag" in response_data


def test_predict_with_invalid_data():
    """
    Tests the /predict endpoint with an invalid payload (missing a field).
    FastAPI should automatically return a 422 error.
    """
    invalid_patient_data = {
        "age": 0.02, "sex": -0.044, "bmi": 0.06, "bp": -0.03,
        "s1": -0.02, "s2": 0.03, "s3": -0.02, "s4": 0.02,
        "s5": 0.02
        # "s6" is missing
    }
    
    response = client.post("/predict", json=invalid_patient_data)
    
    assert response.status_code == 422
