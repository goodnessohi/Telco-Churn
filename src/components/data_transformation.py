import os
import sys
from dataclasses import dataclass

import numpy as np
import dill
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class CustomReplacerConfig:
    columns_to_replace: list = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']

@dataclass
class CustomReplacer(BaseEstimator, TransformerMixin):
    config: CustomReplacerConfig = CustomReplacerConfig()

    def fit(self, df, y=None):
        return self

    def transform(self, df):
        df = df.copy()
        df['MultipleLines'] = df['MultipleLines'].replace('No phone service', 'No')
        for column in self.config.columns_to_replace:
            df[column] = df[column].replace('No internet service', 'No')
        return df

    def fit_transform(self, df, y=None):
        return self.fit(df, y).transform(df)



@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

@dataclass
class DataTransformation:
    data_transformation_config: DataTransformationConfig = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''This is responsible for data transformation '''
        try:
            numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
            categorical_columns = ['customerID', 'gender', 'Partner', 'Dependents', 'PhoneService', 
                                   'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                                   'PaperlessBilling', 'PaymentMethod', 'Churn']
            logging.info('loaded the categorical and numerical columns into the respective variables')

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    
                ]
                    )
            logging.info(f'Numerical pipeline created')
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("label_encoder",LabelEncoder()),
                    
                ]
            )


            preprocessor= ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline,numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:

            raise CustomException(sys)
        
    def initiate_data_transformation(self, train_path, test_path):

        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f'Read train and test data completed')

            logging.info(f'Obtaining preprocessing object')

            preprocessing_obj=self.get_data_transformer_object()

            logging.info('Target column is being separated in df for transformation to occur')
            target_column_name = 'Churn'
            redundant_column_name = 'customerID'

            input_feature_train_df = train_df.drop(columns=[target_column_name, redundant_column_name])
            input_feature_test_df = test_df.drop(columns=[target_column_name, redundant_column_name])
            target_feature_test_df = test_df[target_column_name]
            target_feature_train_df = train_df[target_column_name]

            custom_replacer = CustomReplacer()
            train_data = custom_replacer.fit_transform(input_feature_train_df)
            test_data = custom_replacer.fit_transform(input_feature_test_df)

            preprocessing_obj = self.get_data_transformer_object()
            train_arr = preprocessing_obj.fit_transform(train_data)
            test_arr = preprocessing_obj.transform(test_data)

            
            logging.info(f"Preprocessing completed...")

            save_obj(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor obj saved')
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(f'Error in data ingestion: {str(e)}')
            raise CustomException(sys)
        