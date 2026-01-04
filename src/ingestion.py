import os
import sys
import pandas as pd
import numpy as np
import requests
from io import StringIO
from pathlib import Path
from src.logger import logger
from src.exception import CustomException

class DataIngestion:
    def __init__(self):
        self.raw_data_path = Path("data/raw/heart_raw.csv")
        self.processed_data_path = Path("data/processed/heart_cleaned.csv")
        
    def initiate_data_ingestion(self):
        logger.info("Starting Data Ingestion component")
        try:
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
            column_names = [
                'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
                'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
            ]

            # Download data
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            df = pd.read_csv(StringIO(response.text), names=column_names, na_values='?')
            logger.info("Dataset downloaded successfully from UCI repository")

            # Save Processed file
            os.makedirs(self.raw_data_path.parent, exist_ok=True)
            df.to_csv(self.raw_data_path, index=False)
            logger.info(f"Downloaded Raw data saved to {self.raw_data_path}")

            # Basic Cleaning
            df['target'] = df['target'].apply(lambda x: 1 if x > 0 else 0)
            
            # Handle Missing Values (Imputation)
            for col in df.columns:
                if df[col].isnull().any():
                    val = df[col].median() if df[col].dtype != 'O' else df[col].mode()[0]
                    df[col] = df[col].fillna(val)
            
            # Save Processed file
            os.makedirs(self.processed_data_path.parent, exist_ok=True)
            
            df.to_csv(self.processed_data_path, index=False)
            logger.info(f"Processed data saved to {self.processed_data_path}")

            return self.processed_data_path

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()