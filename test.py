from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

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