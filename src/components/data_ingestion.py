import sys
import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from dataclasses import dataclass

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.exception import CustomException
from src.logger import logging

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig


from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig


@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated")
        try:
            # Read the data from the given path
            df = pd.read_csv('notebook/Dataset/Telco Churn Data.csv')
            logging.info('Dataset read into pandas')

            os.makedirs(os.path.dirname(self.data_ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info('Data saved to csv')

            logging.info("Stratified train-test split initiated")
            split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=64)
            for train_index, test_index in split.split(df, df["Churn"]):
                strat_train_set = df.loc[train_index]
                strat_test_set = df.loc[test_index]

            strat_train_set.to_csv(self.data_ingestion_config.train_data_path, index=False)
            strat_test_set.to_csv(self.data_ingestion_config.test_data_path, index=False)
            logging.info('Train and test datasets created and saved in artifacts')
            logging.info(f"Train dataset shape: {strat_train_set.shape}")
            logging.info(f"Test dataset shape: {strat_test_set.shape}")

            return (
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            ) 

        except Exception as e:
            logging.error(f'Error in data ingestion: {str(e)}')
            raise CustomException(f"Error in data ingestion: {e}")

if __name__ == "__main__":
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, y_train, y_test, preprocessor_obj_file_path = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    model_trainer.initiate_model_training(train_arr, test_arr, y_train, y_test, preprocessor_obj_file_path)