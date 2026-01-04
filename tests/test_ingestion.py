import os
import pytest
from src.ingestion import DataIngestion

def test_data_ingestion_files():
    ingestor = DataIngestion()
    ingestor.initiate_data_ingestion()
    
    assert os.path.exists("data/raw/heart_raw.csv")
    assert os.path.exists("data/processed/heart_cleaned.csv")

def test_data_ingestion_columns():
    import pandas as pd
    df = pd.read_csv("data/processed/heart_cleaned.csv")
    assert "target" in df.columns
    assert df["target"].isin([0, 1]).all()