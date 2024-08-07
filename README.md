# Telco Churn Prediction with SHAP

Welcome to the Telco Churn Prediction project! This project leverages machine learning and SHAP (SHapley Additive exPlanations) to predict customer churn for a telecommunications company and provide insights into the factors influencing churn.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Training](#model-training)
- [SHAP Analysis](#shap-analysis)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## Project Overview

The Telco Churn Prediction project aims to predict whether a customer is likely to churn (leave the company) based on various customer attributes and service usage patterns. The project utilizes a machine learning model to make predictions and SHAP values to explain the model's decisions.

## Features

- **Churn Prediction:** Predicts whether a customer will churn based on input data.
- **User Input:** Provides a user-friendly interface for inputting customer data.
- **SHAP Analysis:** Visualizes feature importance and explains predictions using SHAP values.
- **Interactive Plots:** Displays SHAP summary plots and individual customer analysis plots.

## Installation

To run the Telco Churn Prediction project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/yourusername/Telco-Churn-Prediction.git
   cd Telco-Churn-Prediction

 **Create a virtual environment**
conda create --name telco-churn-venv

 **Activate the Virtual environment on Windows**
 conda activate venv

 **Install required Package**
 pip install -r requirements.txt

 **Usage**
To use the Telco Churn Prediction app, follow these steps:
1. Run the Streamlit app
    streamlit run app.py
2. Input Customer Data
    Use the user-friendly interface to input customer data such as gender, senior citizen status, service usage, and payment details.
3. Predict Churn
    Click the "Predict Churn" button to see whether the customer is likely to churn and the associated probability.
4. Analyze Features
    Explore feature importance and individual customer explanations using the SHAP plots.

**Project Structure**
Telco-Churn-Prediction/
├── artifacts/
│   ├── model.pkl
│   ├── preprocessor.pkl
│   └── train.csv
├── src/
│   ├── components/
│   │   ├── predict_pipeline.py
│   │   └── data_preprocessing.py
│   ├── __init__.py
│   └── utils.py
├── app.py
├── requirements.txt
└── README.md
artifacts/: Contains the trained model, preprocessor, and sample data for SHAP analysis.
    src/components/: Contains the code for prediction pipeline and data preprocessing.
    app.py: The main Streamlit app file.
    requirements.txt: Lists the required Python packages for the project.

Model Training
The model is trained using the Telco customer churn dataset, which includes features such as customer demographics, account information, and service usage patterns. The preprocessing steps and model training are conducted using scikit-learn.

To train the model, follow these steps:

Prepare the data:

Use the data preprocessing script to clean and preprocess the dataset.

Train the model:

Run the model training script to fit the machine learning model.

Save the model and preprocessor:

Save the trained model and preprocessor as artifacts for use in the prediction pipeline.

**SHAP Analysis**
SHAP values provide insights into the model's predictions by quantifying the contribution of each feature. The project includes the following SHAP analyses:

SHAP Summary Plot: Visualizes the overall feature importance across all samples.
Customer-Specific Analysis: Explains individual customer predictions using SHAP force plots.
**Contributing**
Contributions to this project are welcome! If you have suggestions, bug reports, or feature requests, please open an issue or submit a pull request.

**License**
This project is licensed under the MIT License.

**Contact**
For questions or inquiries, please contact Goodness Obaika.
