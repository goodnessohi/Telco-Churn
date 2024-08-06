import os
import sys

from dataclasses import dataclass

from catboost import CatBoostClassifier, Pool
from xgboost import XGBClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.metrics import (
    accuracy_score, classification_report, recall_score, confusion_matrix,
    roc_auc_score, precision_score, f1_score, roc_curve, auc
)

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj, evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    logging.info('ModelTrainerConfig is created')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_training(self, train_arr, test_arr, y_train, y_test, preprocessor_obj_file_path):
        try:
            logging.info('Data training initiated')
            X_train, X_test = train_arr, test_arr
            
            logging.info(f"Train array shape before reshape: {train_arr.shape}")
            logging.info(f"Test array shape before reshape: {test_arr.shape}")
            logging.info(f"y_train shape before reshape: {y_train.shape}")
            logging.info(f"y_test shape before reshape: {y_test.shape}")

            X_train = X_train.reshape(X_train.shape[1], X_train.shape[2])
            X_test = X_test.reshape(X_test.shape[1], X_test.shape[2])
            
            logging.info(f"Train array shape after: {train_arr.shape}")
            logging.info(f"Test array shape after: {test_arr.shape}")
            logging.info(f"y_train shape after: {y_train.shape}")
            logging.info(f"y_test shape after: {y_test.shape}")

            # Convert 'Churn' column to numeric
            y_train = y_train.map({'Yes': 1, 'No': 0})
            y_test = y_test.map({'Yes': 1, 'No': 0})
            
            #Define models
            models = {
                "CatBoostClassifier": CatBoostClassifier(iterations=100, verbose=False),
                "Random Forest": RandomForestClassifier(),
                "SVM": SVC(probability=True),
                "Logistic Regression": LogisticRegression(max_iter=1000)

            }

            params = {
            "Random Forest": {'n_estimators': [10, 50, 100]},
            "LogisticRegression": {},
            "SVM": {'C': [0.1, 1, 10]},
            'CatBoostClassifier': {'learning_rate': [0.01, 0.1], 'depth': [4, 6, 8]},
            'XGBClassifier': {'learning_rate': [0.01, 0.1], 'max_depth': [3, 6, 9]}
                }
            
            best_models = {}
                
            for model_name in models:
                logging.info(f'Tuning hyperparameters for {model_name}')
                grid_search = GridSearchCV(
                    models[model_name],
                    params[model_name],
                    cv=3,
                    n_jobs=-1,
                    scoring='accuracy',
                    error_score='raise'
        )
                grid_search.fit(X_train, y_train)

                #Get best model
                best_models[model_name] = grid_search.best_estimator_

            # Evaluate the best models
            model_report = evaluate_models(X_train, y_train, X_test, y_test, best_models)
            logging.info('Model evaluation is being carried out...')
    
            # Get the best model score from the evaluation report
            best_model_score = max(model_report.values())
            logging.info('Best model score is being obtained...')
    
            # Get the best model name from the evaluation report
            best_model_name = max(model_report, key=model_report.get)
            best_model = best_models[best_model_name]
    
            # Check if the best model score is above the threshold
            if best_model_score < 0.6:
                raise CustomException(f"No best model as the model scores are less than 60%")
    
            logging.info('Best model found on training and test datasets')
    
            # Print the best model and its score
            logging.info(f'Best model: {best_model_name} with a score of {best_model_score:.2f}')

            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            logging.info('Trained model has been saved.')

              # Assuming best_model is obtained from the previous steps
            predicted = best_model.predict(X_test)

            # Calculate and log various classification metrics
            accuracy = accuracy_score(y_test, predicted)
            precision = precision_score(y_test, predicted, average='weighted')
            recall = recall_score(y_test, predicted, average='weighted')
            f1 = f1_score(y_test, predicted, average='weighted')

            logging.info('Trained model is used to make predictions on test data.')
            logging.info(f'Accuracy: {accuracy:.2f}')
            logging.info(f'Precision: {precision:.2f}')
            logging.info(f'Recall: {recall:.2f}')
            logging.info(f'F1 Score: {f1:.2f}')

            # Print a detailed classification report
            logging.info('Classification Report:\n' + classification_report(y_test, predicted))

            # Log the confusion matrix
            conf_matrix = confusion_matrix(y_test, predicted)
            logging.info(f'Confusion Matrix:\n{conf_matrix}')

            # Return the accuracy score or another metric of choice
            return accuracy
   
        except Exception as e:
            raise CustomException(str(e))
