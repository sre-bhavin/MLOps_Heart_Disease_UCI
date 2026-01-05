from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

import logging
from datetime import datetime
from prometheus_fastapi_instrumentator import Instrumentator

# Setup logging configuration
logging.basicConfig(
    filename='api_activity.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Initialize FastAPI
app = FastAPI(title="Heart Disease Prediction API")

# start tracking metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

@app.on_event("startup")
async def expose_metrics():
    instrumentator.expose(app)

# Path to artifacts
MODEL_PATH = Path("models/best_model.pkl")
PREPROCESSOR_PATH = Path("models/preprocessor.pkl")

# Load model and preprocessor once at startup
if not MODEL_PATH.exists() or not PREPROCESSOR_PATH.exists():
    raise RuntimeError("Model or Preprocessor file not found. Run the DVC pipeline first.")

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

# Define the input schema using Pydantic
class PatientData(BaseModel):
    age: int
    sex: int
    cp: int
    trestbps: int
    chol: int
    fbs: int
    restecg: int
    thalach: int
    exang: int
    oldpeak: float
    slope: int
    ca: int
    thal: int

@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running. Visit /docs for Swagger UI."}

@app.post("/predict")
def predict(data: PatientData):
    try:
        # 1. Convert input JSON to DataFrame
        input_df = pd.DataFrame([data.model_dump()])
        
        # 2. Preprocess the data using the saved transformer
        transformed_data = preprocessor.transform(input_df)
        
        # 3. Get Prediction and Probability
        prediction = model.predict(transformed_data)[0]
        probability = model.predict_proba(transformed_data)[0]
        confidence = float(np.max(probability))

        # Log the activity
        logging.info(f"Request: {data.model_dump()} | Prediction: {prediction} | Confidence: {confidence}")

        return {
            "prediction": int(prediction),
            "status": "Positive" if prediction == 1 else "Negative",
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        logging.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))