from main import LR_class
import pytest


@pytest.fixture
def lr_manager():
    """Create a fresh instance of LR_class before each test """
    return LR_class()

def test_load_data(lr_manager):
    lr_manager._load_data()
    dataframe = lr_manager.df
    feature_cols = lr_manager.X
    target_col = lr_manager.y

    assert dataframe is not None
    assert feature_cols is not None
    assert target_col is not None

def test_predict(lr_manager):
    prediction = lr_manager.predict()
    assert prediction is not None

def test_input_predict(lr_manager):
    percentage = 0.7
    result = lr_manager.predict(percentage)
    result = int(result *100) / 100
    assert result != 53.12
