import os
import sys
import dill
import pickle
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
    
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)

def save_obj(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
    except Exception as e:
        print(f"Error: {str(e)}")
        raise CustomException(str(e))


def evaluate_models(X_train, y_train, X_test, y_test, models):
    report = {}
    
    for model_name, model in models.items():
        try:
            # Train the model
            model.fit(X_train, y_train)
            logging.info(f'{model_name} training completed.')

            # Predict the training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            logging.info(f'{model_name} prediction completed.')

            # Calculate various metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)

            # Store metrics in the report
            report[model_name] = {
                'accuracy': accuracy,
                'recall': recall,
            }
            logging.info(f'{model_name} evaluation completed.')
        
        except Exception as e:
            logging.error(f"Error evaluating model {model_name}: {str(e)}")
    
    return report

def find_best_model(model_report, accuracy_weight=0.5, recall_weight=0.5):
    # Initialize variables to store the best model name and score
    best_model_name = None
    best_model_score = -1

    for model_name, metrics in model_report.items():
        # Calculate weighted score
        weighted_score = (accuracy_weight * metrics['accuracy']) + (recall_weight * metrics['recall'])
        
        # Update best model if the current one is better
        if weighted_score > best_model_score:
            best_model_name = model_name
            best_model_score = weighted_score

    return best_model_name, best_model_score



def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)