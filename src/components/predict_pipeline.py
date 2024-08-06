import pickle
import numpy as np
import shap

class ChurnPredictor:
    def __init__(self, model_path, preprocessor_path):
        # Load the trained model
        with open(model_path, 'rb') as model_file:
            self.model = pickle.load(model_file)
        
        # Load the preprocessor
        with open(preprocessor_path, 'rb') as preprocessor_file:
            self.preprocessor = pickle.load(preprocessor_file)
        
        # Initialize the SHAP explainer
        self.explainer = shap.Explainer(self.model, self.preprocessor.transform)
        self.shap_values = None

    def predict(self, input_data):
        # Preprocess the input data
        processed_data = self.preprocessor.transform(np.array(input_data).reshape(1, -1))
        
        # Make predictions
        prediction = self.model.predict(processed_data)
        probability = self.model.predict_proba(processed_data)

        return prediction[0], probability[0][1]  # Return class and probability

    def calculate_shap_values(self, data):
        # Preprocess the data for SHAP values calculation
        processed_data = self.preprocessor.transform(data)
        # Calculate SHAP values
        self.shap_values = self.explainer(processed_data)
        return self.shap_values

    def feature_importance_summary(self, data):
        # Calculate SHAP values if not done
        if self.shap_values is None:
            self.calculate_shap_values(data)

        # Summarize feature importance using SHAP values
        shap.summary_plot(self.shap_values, self.preprocessor.transform(data), show=False)
        return shap.plots._force.force_plot(self.explainer.expected_value[0], self.shap_values[0][0], data.iloc[0])
