import pytest
from src.model_evaluation import ModelEvaluator
from pathlib import Path
import mlflow

import pytest
from mlflow.tracking import MlflowClient

@pytest.fixture(scope="session", autouse=True)
def ensure_experiment():
    client = MlflowClient()
    name = "Heart_Disease_Classification"
    if client.get_experiment_by_name(name) is None:
        client.create_experiment(name)

@pytest.fixture(scope="session", autouse=True)
def ensure_model_version():
    client = MlflowClient()
    reg_name = "heart-disease-model"

    # Create model if missing
    try:
        client.get_registered_model(reg_name)
    except Exception:
        client.create_registered_model(reg_name)

    # Create a model version if none exist
    versions = client.search_model_versions(f"name='{reg_name}'")
    if not versions:
        # Log a dummy artifact to register
        with mlflow.start_run():
            mlflow.log_param("init", True)
            mlflow.log_metric("accuracy", 0.80)
            mlflow.log_artifact("README.md")  # ensure some artifact exists
            run_id = mlflow.active_run().info.run_id
        client.create_model_version(
            name=reg_name,
            source=f"mlruns/{mlflow.active_run().info.experiment_id}/{run_id}/artifacts",
            run_id=run_id
        )

    # Transition the latest version to Staging
    latest_version = client.search_model_versions(f"name='{reg_name}'")[0]
    client.transition_model_version_stage(
        name=reg_name,
        version=latest_version.version,
        stage="Staging"
    )



@pytest.fixture(scope="session", autouse=True)
def ensure_run_in_experiment():
    client = MlflowClient()
    exp_name = "my-experiment"
    experiment = client.get_experiment_by_name(exp_name)
    if experiment is None:
        exp_id = client.create_experiment(exp_name)
    else:
        exp_id = experiment.experiment_id

    # Start and log a run if none exist
    runs = client.search_runs([exp_id])
    if not runs:
        with mlflow.start_run(experiment_id=exp_id):
            mlflow.log_metric("accuracy", 0.85)
            mlflow.log_param("model_type", "baseline")


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