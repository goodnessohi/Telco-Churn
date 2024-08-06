import pandas as pd
import os
import sys
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

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
            categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                                   'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup', 
                                   'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract', 
                                   'PaperlessBilling', 'PaymentMethod']
            logging.info('Loaded the categorical and numerical columns into the respective variables')

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                ]
            )
            logging.info(f'Numerical pipeline created....Proceeding to categorical')
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown='ignore', sparse=False)),  # Ensure dense output
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException("Error in creating preprocessor object", sys.exc_info())

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Read CSV files
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info('Read train and test data completed')

            # Check for the 'Churn' column
            if 'Churn' not in train_df.columns or 'Churn' not in test_df.columns:
                raise ValueError("The 'Churn' column is not present in the training or test dataset.")

            # Define target and redundant columns
            target_column_name = 'Churn'
            redundant_column_name = 'customerID'

            # Drop the redundant columns
            X_train = train_df.drop(columns=[redundant_column_name, target_column_name], axis=1)
            y_train = train_df[target_column_name]
            X_test = test_df.drop(columns=[redundant_column_name, target_column_name], axis=1)
            y_test = test_df[target_column_name]

            # Convert TotalCharges to numeric, forcing errors to NaN
            X_train['TotalCharges'] = pd.to_numeric(X_train['TotalCharges'], errors='coerce')
            X_test['TotalCharges'] = pd.to_numeric(X_test['TotalCharges'], errors='coerce')

            # Replace 'No internet service' with 'No' in specified columns
            columns_to_replace = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
            for column in columns_to_replace:
                if column in X_train.columns:
                    X_train[column] = X_train[column].replace('No internet service', 'No')
                    X_test[column] = X_test[column].replace('No internet service', 'No')
                else:
                    logging.warning(f"Column {column} not found in DataFrame.")

            # Replace 'No phone service' with 'No' in the 'MultipleLines' column
            if 'MultipleLines' in X_train.columns:
                X_train['MultipleLines'] = X_train['MultipleLines'].replace('No phone service', 'No')
                X_test['MultipleLines'] = X_test['MultipleLines'].replace('No phone service', 'No')
            else:
                logging.warning("Column 'MultipleLines' not found in DataFrame.")

            logging.info('Replacement done in categorical columns')

            # Log data types before transformation
            logging.info(f'Data types before transformation:\n{X_train.dtypes}')

            # Get the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Transform the data using the preprocessing pipeline
            X_train_arr = preprocessing_obj.fit_transform(X_train)
            X_test_arr = preprocessing_obj.transform(X_test)

            # Log transformed data shape and sample
            logging.info(f'X_train_arr shape after transformation: {X_train_arr.shape}')
            logging.info(f'X_test_arr shape after transformation: {X_test_arr.shape}')
            logging.info(f'First row of X_train_arr:\n{X_train_arr[0]}')
            logging.info(f'First row of X_test_arr:\n{X_test_arr[0]}')

            # Check if all columns are numeric
            if not np.issubdtype(X_train_arr.dtype, np.number):
                non_numeric_train_cols = [i for i, col in enumerate(X_train_arr.T) if not np.issubdtype(col.dtype, np.number)]
                logging.error(f'Non-numeric columns in transformed training data: {non_numeric_train_cols}')
                raise ValueError("Not all columns in the training data are numeric after transformation")

            if not np.issubdtype(X_test_arr.dtype, np.number):
                non_numeric_test_cols = [i for i, col in enumerate(X_test_arr.T) if not np.issubdtype(col.dtype, np.number)]
                logging.error(f'Non-numeric columns in transformed test data: {non_numeric_test_cols}')
                raise ValueError("Not all columns in the test data are numeric after transformation")

            logging.info('Preprocessing completed')
            logging.info('Attempting to save preprocessor obj')

            # Save the preprocessor object
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessor obj saved')

            return (
                X_train_arr,
                X_test_arr,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            logging.error(f'Error in data transformation: {str(e)}')
            raise CustomException(f"Error occurred during data transformation: {str(e)}")

