from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")


def test_data_ingestion_output():
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    assert train_data == "artifacts/train.csv"
    assert test_data == "artifacts/test.csv"


def test_data_transformation_output():
    obj = DataTransformation()
    preprocessor = obj.get_data_transformer_object()
    assert preprocessor is not None

    train_path = "artifacts/train.csv"
    test_path = "artifacts/test.csv"

    train_arr, test_arr, preprocessor_path = obj.initiate_data_transformation(train_path, test_path)

    assert train_arr is not None
    assert test_arr is not None
    assert preprocessor_path is not None

    return train_arr, test_arr


def test_model_output():
    obj = ModelTrainer()
    train_arr, test_arr = test_data_transformation_output()
    predicted, r2_score_value = obj.initiate_model_trainer(train_arr, test_arr)
    print(predicted, r2_score_value)

    assert obj.model_trainer_config.trained_model_file_path == "artifacts/model.pkl"
    assert predicted.isnumeric()
    assert r2_score_value.isnumeric() and r2_score_value <= 1 and r2_score_value >= 0

if __name__ == "__main__":
    test_model_output() 