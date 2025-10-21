import sys
import os
from dataclasses import dataclass

# Modelling
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression, Ridge,Lasso
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    # Where the best model will be saved
    trained_model_file_path = os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1], # All columns except last = Features
                train_array[:,-1], # Only last column = Target
                test_array[:,:-1],
                test_array[:,-1]
            )

            # Suggested improvement: organize models for streamlined evaluation and future hyperparameter tuning
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest Regressor": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(), 
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor()
            }

            # Define hyperparameter grids for each model
            params = {
                "Linear Regression": {},
                
                "Lasso": {
                    'alpha': [0.1, 0.5, 1.0, 2.0, 5.0],
                    'max_iter': [1000, 2000, 5000]
                },
                
                "Random Forest Regressor": {
                    'n_estimators': [50, 100, 200, 500],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                
                "XGBRegressor": {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.8, 0.9, 1.0]
                },
                
                "Decision Tree": {
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                
                "K-Neighbors Regressor": {
                    'n_neighbors': [3, 5, 7, 9, 11],
                    'weights': ['uniform', 'distance'],
                    'algorithm': ['auto', 'ball_tree', 'kd_tree']
                }
            }

            model_report:dict = evaluate_models(X_train = X_train, y_train = y_train, X_test = X_test, y_test = y_test,
                                             models = models, params = params)
            '''
            # To get best model score from dict
            #  Refer Example Working 1
            best_model_score = max(sorted(model_report.values()))

            # To get best model name from dict
            # Refer Example Working 2
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]'''

            # Find the key with maximum value directly
            # Example Working 3
            best_model_name = max(model_report, key=model_report.get)
            # this looks different from the usual dict.get(key) pattern
            # Here, `model_report.get` is passed as a FUNCTION, not called immediately
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]
                        
            # Reject if no model is good enough
            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            # if you want to load a pkl file 
            # preprocessing_obj = 

            # Save the best model as a .pkl file
            save_object(
                file_path = self.model_trainer_config.trained_model_file_path,
                obj = best_model
            )

            # Test the saved model one more time
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test,predicted)
            return r2 # # Return final R² score

        except Exception as e:
            raise CustomException(e,sys)