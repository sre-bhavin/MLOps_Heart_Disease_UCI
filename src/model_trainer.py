import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay

from src.logger import logger
from src.exception import CustomException

class ModelTrainer:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.data_path = self.project_root / "data" / "processed" / "heart_transformed.csv"
        self.model_dir = self.project_root / "models"
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def eval_metrics(self, actual, pred, pred_proba):
        accuracy = accuracy_score(actual, pred)
        precision = precision_score(actual, pred)
        recall = recall_score(actual, pred)
        roc_auc = roc_auc_score(actual, pred_proba)
        return accuracy, precision, recall, roc_auc

    def initiate_model_trainer(self):
        try:
            logger.info("Loading transformed data")
            df = pd.read_csv(self.data_path)
            X = df.drop(columns=['target'])
            y = df['target']

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            models = {
                "Logistic_Regression": {
                    "model": LogisticRegression(max_iter=1000),
                    "params": {
                        "C": [0.1, 1.0, 10.0],
                        "solver": ["liblinear", "lbfgs"]
                    }
                },
                "Random_Forest": {
                    "model": RandomForestClassifier(),
                    "params": {
                        "n_estimators": [50, 100, 200],
                        "max_depth": [None, 10, 20],
                        "min_samples_split": [2, 5]
                    }
                }
            }

            mlflow.set_experiment("Heart_Disease_Classification")

            for model_name, config in models.items():
                with mlflow.start_run(run_name=model_name):
                    logger.info(f"Started tuning and training for: {model_name}")
                    
                    # Hyperparameter Tuning using Cross-Validation
                    gs = GridSearchCV(config["model"], config["params"], cv=5, scoring='accuracy')
                    gs.fit(X_train, y_train)

                    best_model = gs.best_estimator_
                    
                    # Predictions
                    y_pred = best_model.predict(X_test)
                    y_prob = best_model.predict_proba(X_test)[:, 1]

                    # Metrics
                    acc, prec, rec, roc = self.eval_metrics(y_test, y_pred, y_prob)

                    # Log Params and Metrics to MLflow
                    mlflow.log_params(gs.best_params_)
                    mlflow.log_metric("accuracy", acc)
                    mlflow.log_metric("precision", prec)
                    mlflow.log_metric("recall", rec)
                    mlflow.log_metric("roc_auc", roc)

                    # Create and Log Confusion Matrix Plot
                    plt.figure(figsize=(6,6))
                    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
                    plot_path = self.model_dir / f"{model_name}_cm.png"
                    plt.savefig(plot_path)
                    mlflow.log_artifact(str(plot_path))
                    plt.close()

                    # Log the Model
                    mlflow.sklearn.log_model(best_model, "model")
                    
                    logger.info(f"{model_name} training complete. Accuracy: {acc:.4f}")

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.initiate_model_trainer()