import pytest
import os
from pathlib import Path
from src.model_trainer import ModelTrainer

def test_model_trainer_outputs():
    """Test if training produces the expected artifacts."""
    trainer = ModelTrainer()
    
    # Ensure clean state for test
    if os.path.exists("models/Logistic_Regression_cm.png"):
        os.remove("models/Logistic_Regression_cm.png")
        
    trainer.initiate_model_trainer()
    
    assert os.path.exists("models/Logistic_Regression_cm.png")
    assert os.path.exists("models/Random_Forest_cm.png")

def test_eval_metrics_logic():
    """Test the metric calculation helper directly."""
    trainer = ModelTrainer()
    actual = [1, 0, 1, 0]
    pred = [1, 0, 0, 0]
    pred_proba = [0.9, 0.1, 0.2, 0.3]
    
    acc, prec, rec, roc = trainer.eval_metrics(actual, pred, pred_proba)
    
    assert 0 <= acc <= 1
    assert rec == 0.5  # 1 out of 2 positives caught