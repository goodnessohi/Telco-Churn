import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from src.components.predict_pipeline import ChurnPredictor

# Initialize the predictor
predictor = ChurnPredictor(
    model_path="artifacts/model.pkl",
    preprocessor_path="artifacts/preprocessor.pkl",
    sample_data_path="artifacts/train.csv"  # Use train.csv as the sample data for SHAP
)

def main():
    st.title("Telco Churn Prediction with SHAP")

    # Define input fields for user input
    customer_id = st.text_input("Customer ID", "7590-VHVEG")
    gender = st.selectbox("Gender", options=['Female', 'Male'])
    senior_citizen = st.selectbox("Senior Citizen", options=[0, 1])  # Add Senior Citizen
    partner = st.selectbox("Partner", options=['Yes', 'No'])
    dependents = st.selectbox("Dependents", options=['Yes', 'No'])
    phone_service = st.selectbox("Phone Service", options=['Yes', 'No'])
    multiple_lines = st.selectbox("Multiple Lines", options=['No phone service', 'No', 'Yes'])
    internet_service = st.selectbox("Internet Service", options=['DSL', 'Fiber optic', 'No'])
    online_security = st.selectbox("Online Security", options=['No', 'Yes', 'No internet service'])
    online_backup = st.selectbox("Online Backup", options=['No', 'Yes', 'No internet service'])
    device_protection = st.selectbox("Device Protection", options=['No', 'Yes', 'No internet service'])
    tech_support = st.selectbox("Tech Support", options=['No', 'Yes', 'No internet service'])
    streaming_tv = st.selectbox("Streaming TV", options=['No', 'Yes', 'No internet service'])
    streaming_movies = st.selectbox("Streaming Movies", options=['No', 'Yes', 'No internet service'])
    contract = st.selectbox("Contract", options=['Month-to-month', 'One year', 'Two year'])
    paperless_billing = st.selectbox("Paperless Billing", options=['Yes', 'No'])
    payment_method = st.selectbox("Payment Method", options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=1)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=500.0)

    # Compile inputs into a dataframe
    input_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [senior_citizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'PhoneService': [phone_service],
        'MultipleLines': [multiple_lines],
        'InternetService': [internet_service],
        'OnlineSecurity': [online_security],
        'OnlineBackup': [online_backup],
        'DeviceProtection': [device_protection],
        'TechSupport': [tech_support],
        'StreamingTV': [streaming_tv],
        'StreamingMovies': [streaming_movies],
        'Contract': [contract],
        'PaperlessBilling': [paperless_billing],
        'PaymentMethod': [payment_method],
        'tenure': [tenure],
        'MonthlyCharges': [monthly_charges],
        'TotalCharges': [total_charges]
    })

    # Button for making predictions
    if st.button("Predict Churn"):
        prediction, probability = predictor.predict(input_data)
        if prediction == 1:
            st.error(f"Customer is likely to churn. Probability: {probability:.2f}")
        else:
            st.success(f"Customer is unlikely to churn. Probability: {probability:.2f}")

    # Display SHAP Feature Importance Summary
    st.subheader("Feature Importance Summary")
    st.text("This plot shows the importance of each feature in the model.")
    
    # Use sample data from the predictor for SHAP analysis
    shap_values = predictor.calculate_shap_values(pd.read_csv("artifacts/train.csv").drop(columns=['customerID', 'Churn'], errors='ignore'))

    # Plot SHAP Summary
    plt.figure()
    shap.summary_plot(shap_values, predictor.X_background, feature_names=predictor.get_feature_names(), show=False)
    st.pyplot(plt)

    # User-based SHAP Analysis
    st.subheader("Customer-specific SHAP Analysis")
    customer_index = st.number_input("Select Customer Index for SHAP Analysis", min_value=0, max_value=len(predictor.X_background)-1, value=0)

    # Plot SHAP force plot for a specific customer
    customer_data = pd.DataFrame(predictor.X_background[customer_index].reshape(1, -1), columns=predictor.get_feature_names())
    shap.force_plot(predictor.explainer.expected_value, shap_values[customer_index], customer_data, matplotlib=True, show=False)
    st.pyplot(plt)

if __name__ == '__main__':
    main()
