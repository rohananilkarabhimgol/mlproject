# will have common functionalities that the entire project can use
import os 
import sys

import pandas as pd
import numpy as np
import dill
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV


from src.exception import CustomException

def save_object(file_path,obj):
    try:
        # Path: "artifacts/preprocessor.pkl
        dir_path = os.path.dirname(file_path) # artifects
        os.makedirs(dir_path,exist_ok=True) # Creates artifects folder if not exists

        # https://dill.readthedocs.io/en/latest/
        with open(file_path,"wb") as f:
            dill.dump(obj,f)

    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    # Example Working 1
    try:
        report = {}
        
        # Iterate directly through items (cleaner)
        for model_name, model in models.items():  

            # Get hyperparameters for this model
            # Example Working 2
            model_params = params.get(model_name, {})

            # Replace RandomizedSearchCV with:
            gs = GridSearchCV(
                model, 
                model_params, 
                cv=5,
                scoring='r2',
                n_jobs=-1
            )

            # Find Best Parameters
            gs.fit(X_train,y_train)

            # Apply Best Parameters
            model.set_params(**gs.best_params_)

            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_test_pred = model.predict(X_test)
            
            # Get R2 score
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score
            
        return report
        
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path,"rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e,sys)
    