import os
import joblib
from src.transformation import DataTransformation

def test_transformation_output():
    transformer = DataTransformation()
    transformer.initiate_data_transformation()
    
    assert os.path.exists("data/processed/heart_transformed.csv")
    assert os.path.exists("models/preprocessor.pkl")

def test_preprocessor_loading():
    preprocessor = joblib.load("models/preprocessor.pkl")
    assert hasattr(preprocessor, "transform")  # Check if it's a valid sklearn object