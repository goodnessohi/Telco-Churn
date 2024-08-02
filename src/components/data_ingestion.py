import sys
import os
import pandas as pd
import sklearn

from src.exception import CustomException
from src.logger import logging

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path: str=os.path.join('artifacts', 'train.csv')
    test_data_path: str=os.path.join('artifacts', 'test.csv')
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion initiated")
        try:
            # Read the data from the given path
            df = pd.read_csv('notebook/Dataset/Telco Churn Data.csv')
            logging.info('dataset read into pandas')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.data_ingestion_config.raw_data_path, index=False)
            logging.info('data saved to csv')

            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=64)
            train_set.to_csv(self.data_ingestion_config.train_data_path, index=False)
            test_set.to_csv(self.data_ingestion_config.test_data_path, index=False)
            logging.info('train and test datasets created and saved in artifacts')

            

        except:
            pass