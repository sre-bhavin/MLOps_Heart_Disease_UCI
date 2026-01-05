import pytest
from src.model_evaluation import ModelEvaluator
from pathlib import Path
import mlflow

def test_mlflow_experiment_exists():
    """Check if the evaluation script can find the MLflow experiment."""
    evaluator = ModelEvaluator()
    experiment = mlflow.get_experiment_by_name(evaluator.experiment_name)
    assert experiment is not None
    assert experiment.lifecycle_stage == "active"

def test_best_model_export():
    """Verify that the evaluator successfully exports the best_model.pkl file."""
    evaluator = ModelEvaluator()
    dest_path = Path("models/best_model.pkl")
    
    # Trigger evaluation
    evaluator.evaluate_and_register()
    
    assert dest_path.exists()
    assert dest_path.stat().st_size > 0  # Check if file is not empty