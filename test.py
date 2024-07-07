from src.components.data_ingestion import DataIngestion

def test_data_ingestion_output():
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    assert train_data == "artifacts/train.csv"
    assert test_data == "artifacts/test.csv"