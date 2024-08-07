import pandas as pd
import numpy as np
import shap
import joblib

class ChurnPredictor:
    def __init__(self, model_path, preprocessor_path, sample_data_path):
        # Load the model and preprocessor
        self.model = joblib.load(model_path)
        self.preprocessor = joblib.load(preprocessor_path)
        
        # Load and preprocess the sample data for SHAP
        sample_data = pd.read_csv(sample_data_path)
        sample_data = self.preprocess_sample_data(sample_data)
        
        # Transform the sample data
        self.X_background = self.preprocessor.transform(sample_data)
        
        # Initialize SHAP explainer
        self.explainer = shap.LinearExplainer(self.model, self.X_background)
        
        # Store column names for reference
        self.categorical_columns = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                                    'MultipleLines', 'InternetService', 'OnlineSecurity', 
                                    'OnlineBackup', 'DeviceProtection', 'TechSupport', 
                                    'StreamingTV', 'StreamingMovies', 'Contract', 
                                    'PaperlessBilling', 'PaymentMethod']
        
        self.numerical_columns = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']

    def preprocess_sample_data(self, data):
        """
        Preprocesses the sample data for SHAP analysis.
        """
        # Drop unnecessary columns
        data = data.drop(columns=['customerID', 'Churn'], errors='ignore')
        
        # Ensure TotalCharges is numeric, handling any non-numeric values
        data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce')
        
        # Drop rows with NaN values in TotalCharges
        data = data.dropna(subset=['TotalCharges'])
        
        return data

    def predict(self, input_data):
        """
        Predicts churn for a given input data.
        """
        # Preprocess input data
        processed_data = self.preprocessor.transform(input_data)
        
        # Make prediction
        prediction = self.model.predict(processed_data)
        
        # Get prediction probability
        probability = self.model.predict_proba(processed_data)[:, 1]
        
        return prediction[0], probability[0]

    def calculate_shap_values(self, data):
        """
        Calculates SHAP values for a given data.
        """
        # Ensure the data is a DataFrame before transformation
        data = self.preprocess_sample_data(data)
        
        # Transform the data
        processed_data = self.preprocessor.transform(data)
        
        # Calculate SHAP values
        shap_values = self.explainer.shap_values(processed_data)
        
        return shap_values

    def get_feature_names(self):
        """
        Retrieves feature names from the preprocessor.
        """
        # Get feature names from the preprocessor's OneHotEncoder
        cat_pipeline = self.preprocessor.named_transformers_['cat_pipeline']
        ohe = cat_pipeline.named_steps['onehot']
        categorical_feature_names = ohe.get_feature_names_out(self.categorical_columns)
        
        return np.concatenate([self.numerical_columns, categorical_feature_names])
