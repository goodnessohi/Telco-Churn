import os
import sys
import dill
import pickle
import numpy as np
import pandas as pd

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
            
            # Predict the training and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            # Calculate various metrics
            accuracy = accuracy_score(y_test, y_test_pred)
            precision = precision_score(y_test, y_test_pred, average='weighted', zero_division=0)
            recall = recall_score(y_test, y_test_pred, average='weighted', zero_division=0)
            f1 = f1_score(y_test, y_test_pred, average='weighted', zero_division=0)
            
            # Calculate ROC AUC only if y_test has more than one class
            if len(set(y_test)) > 1:
                if hasattr(model, "predict_proba"):
                    y_test_prob = model.predict_proba(X_test)[:, 1]
                else:
                    y_test_prob = model.decision_function(X_test)
                    
                roc_auc = roc_auc_score(y_test, y_test_prob, multi_class='ovr', average='weighted')
            else:
                roc_auc = None
            
            # Confusion Matrix
            conf_matrix = confusion_matrix(y_test, y_test_pred)
            
            # Classification Report
            class_report = classification_report(y_test, y_test_pred, zero_division=0)
            
            # Store all metrics in the report
            report[model_name] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'roc_auc': roc_auc,
                'confusion_matrix': conf_matrix,
                'classification_report': class_report
            }
        
        except Exception as e:
            print(f"Error evaluating model {model_name}: {str(e)}")
    
    return report

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)