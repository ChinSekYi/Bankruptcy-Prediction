"""
Module for data transformation operations including preprocessing and saving the 
preprocessor object.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, FunctionTransformer

# Custom imports
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from src.components.data_cleaning import AsDiscrete, map_class_labels

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))

@dataclass
class DataTransformationConfig:
    """
    Configuration class for data transformation operations.
    """

    preprocessor_ob_file_path = os.path.join("artifacts", "preprocessor.pkl")


class DataTransformation:
    """
    DataTransformation class handles data preprocessing and transformation operations.

    Methods:
    - __init__(): Initializes a DataTransformation instance with default configuration.
    - get_data_transformer_object(): Returns the preprocessing object.
    - initiate_data_transformation(train_path, test_path): Initiates data transformation,
      performs preprocessing on train and test datasets, and saves the preprocessor object.
    """

    def __init__(self):
        """
        Initializes a DataTransformation instance with default configuration.
        """
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Returns the preprocessing object.
        """
        try:
            numerical_columns = list(range(0, 64))
            categorical_columns = 64
            num_pipeline = Pipeline(
                steps=[
                    ("toDiscrete", AsDiscrete()),
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler()),
                ]
            )
            cat_pipeline = Pipeline(
                ("label_mapping", FunctionTransformer(map_class_labels)),
            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    (
                        "num_pipeline",
                        num_pipeline,
                        numerical_columns,
                    ),
                    ("cat_pipeline", cat_pipeline, categorical_columns),
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys) from e

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process.
        Reads train and test datasets, applies preprocessing, and saves the preprocessor object.

        Args:
        - train_path (str): File path to the training dataset.
        - test_path (str): File path to the testing dataset.

        Returns:
        - Tuple: Transformed train and test datasets and the file path of the preprocessor object.
        """
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test completed")
            logging.info("Obtaining preprocessing object")

            preprocessing_obj = self.get_data_transformer_object()

            # numerical_columns = list(range(0, 64))
            target_column_index = 64

            logging.info("Renaming columns")
            column_names = pd.read_csv("src/components/column_names.txt", header=None)
            train_df.columns = list(column_names[0])
            test_df.columns = list(column_names[0])

            input_feature_train_df = train_df.drop(
                train_df.columns[target_column_index], axis=1, inplace=False
            )
            target_feature_train_df = train_df.iloc[:, target_column_index]

            input_feature_test_df = test_df.drop(
                train_df.columns[target_column_index], axis=1, inplace=False
            )
            target_feature_test_df = test_df.iloc[:, target_column_index]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe"
            )

            input_feature_train_arr = preprocessing_obj.fit_transform(
                input_feature_train_df
            )
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")

            save_object(
                file_path=self.data_transformation_config.preprocessor_ob_file_path,
                obj=preprocessing_obj,
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_ob_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys) from e
