import os
import sys
import pandas as pd
import joblib
from pathlib import Path
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from src.logger import logger
from src.exception import CustomException

class DataTransformation:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parent.parent
        self.processed_data_path = self.project_root / "data" / "processed" / "heart_cleaned.csv"
        self.transformed_data_path = self.project_root / "data" / "processed" / "heart_transformed.csv"
        self.preprocessor_obj_file_path = self.project_root / "models" / "preprocessor.pkl"

    def get_data_transformer_object(self):
        """Creates a ColumnTransformer object for scaling and encoding."""
        try:
            # Define columns based on your EDA
            numerical_columns = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
            categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

            # Create pipeline for numerical features
            num_pipeline = StandardScaler()

            # Create pipeline for categorical features
            cat_pipeline = OneHotEncoder(handle_unknown='ignore')

            logger.info(f"Numerical columns: {numerical_columns}")
            logger.info(f"Categorical columns: {categorical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self):
        try:
            logger.info("Reading processed data for transformation")
            df = pd.read_csv(self.processed_data_path)

            target_column_name = "target"
            input_feature_df = df.drop(columns=[target_column_name], axis=1)
            target_feature_df = df[target_column_name]

            logger.info("Applying preprocessing object on training and testing dataframes")
            preprocessing_obj = self.get_data_transformer_object()
            
            # Transform features
            input_feature_arr = preprocessing_obj.fit_transform(input_feature_df)

            # Combine transformed features and target
            # Note: OneHotEncoder returns a sparse matrix or dense array depending on settings
            transformed_df = pd.DataFrame(input_feature_arr)
            transformed_df[target_column_name] = target_feature_df.values

            # Save transformed data
            transformed_df.to_csv(self.transformed_data_path, index=False)
            
            # Save the preprocessor for inference later
            os.makedirs(os.path.dirname(self.preprocessor_obj_file_path), exist_ok=True)
            joblib.dump(preprocessing_obj, self.preprocessor_obj_file_path)

            logger.info(f"Transformation complete. Saved to {self.transformed_data_path}")
            return str(self.transformed_data_path)

        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    obj = DataTransformation()
    obj.initiate_data_transformation()