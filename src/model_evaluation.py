import os
import sys
import shutil
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient
from src.logger import logger
from src.exception import CustomException

class ModelEvaluator:
    def __init__(self):
        self.experiment_name = "Heart_Disease_Classification"
        # We select the best model based on Recall to minimize False Negatives
        self.target_metric = "metrics.recall" 
        self.project_root = Path(__file__).resolve().parent.parent
        self.destination_path = self.project_root / "models" / "best_model.pkl"

    def evaluate_and_register(self):
        try:
            logger.info("Starting model evaluation and registration process")
            
            # 1. Connect to MLflow
            client = MlflowClient()
            experiment = client.get_experiment_by_name(self.experiment_name)
            
            if not experiment:
                raise Exception(f"Experiment {self.experiment_name} not found!")

            # 2. Search for the best run in the current experiment based on Recall
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=[f"{self.target_metric} DESC"],
                max_results=1
            )

            if not runs:
                raise Exception("No runs found in the experiment.")

            best_run = runs[0]
            best_run_id = best_run.info.run_id
            best_model_name = best_run.data.tags.get("mlflow.runName", "Unknown_Model")
            best_metric_value = best_run.data.metrics.get("recall")

            logger.info(f"Winner Model: {best_model_name} with Recall: {best_metric_value:.4f}")

            # 3. Export the Winning Model to .pkl for Flask/Docker use
            logger.info(f"Exporting the best model to {self.destination_path}")
            
            # Download model artifacts to a temporary directory
            local_path = mlflow.artifacts.download_artifacts(run_id=best_run_id, artifact_path="model")
            
            # Scikit-learn models logged in MLflow contain a 'model.pkl' inside the 'model' folder
            source_pkl = Path(local_path) / "model.pkl"
            
            # Copy to our project's models folder
            shutil.copy(source_pkl, self.destination_path)
            logger.info("Successfully exported best_model.pkl")

            # 4. Register the model in the MLflow Model Registry
            model_uri = f"runs:/{best_run_id}/model"
            reg_name = "HeartDiseaseClassifier"
            
            result = mlflow.register_model(model_uri, reg_name)
            logger.info(f"Successfully registered model version {result.version} to Registry.")

            # 5. Transition to "Staging"
            client.transition_model_version_stage(
                name=reg_name,
                version=result.version,
                stage="Staging"
            )
            logger.info(f"Model version {result.version} promoted to 'Staging' stage.")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    evaluator = ModelEvaluator()
    evaluator.evaluate_and_register()