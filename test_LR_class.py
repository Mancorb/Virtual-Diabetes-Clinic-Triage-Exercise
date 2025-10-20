from LR_class import LR_class
import pytest
from app import app
from fastapi.testclient import TestClient

client = TestClient(app)

@pytest.fixture
def lr_manager():
    return LR_class()

def test_load_data(lr_manager):
    """Test that the data loads correctly"""
    lr_manager._load_data()  # assuming your class has this method
    dataframe = lr_manager.df
    feature_cols = lr_manager.X
    target_col = lr_manager.y

    assert dataframe is not None
    assert feature_cols is not None
    assert target_col is not None


def test_health_endpoint():
    """Test the /health endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "ok"
    assert "model_version" in json_data


def test_predict_endpoint():
    """Test the /predict endpoint with example input"""
    input_data = {
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
    response = client.post("/predict", json=input_data)
    assert response.status_code == 200
    json_data = response.json()
    assert "prediction" in json_data
    assert isinstance(json_data["prediction"], float)