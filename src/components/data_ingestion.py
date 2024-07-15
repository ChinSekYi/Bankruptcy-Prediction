"""
data_ingestion.py

This module handles the configuration and ingestion of training, testing, and raw data files.
It defines a DataIngestionConfig class for setting default paths and imports necessary components
for data transformation, model training, exception handling, logging, and printing bankruptcy 
outcomes.

Classes:
- DataIngestionConfig: Configuration class for data paths.

Functions:
- None

Note:
- This module imports components from src.components for data transformation (DataTransformation,
  DataTransformationConfig), model training (ModelTrainer, ModelTrainerConfig), exception handling
  (CustomException), logging (logging), and utility functions for printing bankruptcy outcomes
  (print_bankruptcy_outcome).
- It uses standard library modules such as os, sys, dataclasses, and pathlib, as well as third-party
  modules like sklearn.model_selection.train_test_split.
"""

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.exception import CustomException
from src.logger import logging
from src.utils import print_bankruptcy_outcome

project_root = Path(__file__).resolve().parents[2]
sys.path.append(str(project_root))


@dataclass
class DataIngestionConfig:
    """
    Configuration class for data paths used in data ingestion processes.

    Attributes:
    - train_data_path (str): Path to training data file.
    - test_data_path (str): Path to testing data file.
    - raw_data_path (str): Path to raw data file.
    """

    train_data_path: str = os.path.join("artifacts", "train.csv")
    test_data_path: str = os.path.join("artifacts", "test.csv")
    raw_data_path: str = os.path.join("artifacts", "data.csv")


class DataIngestion:
    """
    DataIngestion class handles the ingestion of raw data,
    performs train-test split, and saves the split datasets.

    Attributes:
    - ingestion_config (DataIngestionConfig): Configuration object for data paths.

    Methods:
    - __init__(): Initializes a new instance of DataIngestion with default configuration.
    - initiate_data_ingestion(): Initiates the data ingestion process, reads a CSV file,
      performs train-test split, and saves the datasets to specified paths.
    """

    def __init__(self):
        """
        Initializes a DataIngestion instance with default configuration.
        """
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Initiates the data ingestion process.

        Reads a CSV file from a specified path, performs train-test split,
        and saves the split datasets to predefined paths.

        Returns:
        - Tuple containing paths to the training and testing datasets.

        Raises:
        - CustomException: If an error occurs during data ingestion process.

        Usage:
        >>> train_data_path, test_data_path = ingestion.initiate_data_ingestion()
        """
        logging.info("Entered the data ingestion method or component")
        try:
            df = pd.read_csv(
                "notebook/data/3year.csv"
            )  # change this to read from other database
            logging.info("Read the dataset as dataframe")

            os.makedirs(
                os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True
            )

            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.3, random_state=42)

            train_set.to_csv(
                self.ingestion_config.train_data_path, index=False, header=True
            )

            test_set.to_csv(
                self.ingestion_config.test_data_path, index=False, header=True
            )

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,
            )

        except Exception as e:
            raise CustomException(e, sys) from e


if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
        train_data, test_data
    )

    modeltrainer = ModelTrainer()
    pred_result, r2_score = modeltrainer.initiate_model_trainer(train_arr, test_arr)

    print(f"Prediction result (0-1): {pred_result}")
    print(len(pred_result))
    print(print_bankruptcy_outcome(pred_result))
    print(f"r2 score: {r2_score}")
