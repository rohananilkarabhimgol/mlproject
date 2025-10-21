import sys  # Provides access to system-specific parameters and functions (e.g., for exception handling or path management)
import os
from dataclasses import dataclass  # Simplifies class creation by auto-generating special methods like __init__ and __repr__

import numpy as np  # Fundamental package for numerical computations and array operations
import pandas as pd  # Powerful data manipulation and analysis library, especially for tabular data

from sklearn.compose import ColumnTransformer  # Enables applying different preprocessing steps to different columns in a pipeline
from sklearn.impute import SimpleImputer  # Handles missing values by replacing them with a specified strategy (mean, median, etc.)
from sklearn.pipeline import Pipeline  # Chains multiple preprocessing and modeling steps into a single workflow
from sklearn.preprocessing import OneHotEncoder, StandardScaler  # OneHotEncoder converts categorical variables to binary format; StandardScaler normalizes numerical features

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts","preprocessor.pkl")

# https://medium.com/@pujalabhanuprakash/understanding-the-difference-between-column-transformation-and-pipeline-in-scikit-learn-4b7fb252b52e
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    # 1. CREATE transformer (blueprint)
    def get_data_transformer_object(self):
        '''
        # This function handles various types of data transformations.
        # It returns a PREPROCESSOR OBJECT that knows:
        # - How to process numerical columns
        # - How to encode categorical columns
        # - Which columns are numerical and which are categorical

        '''
        try:
            numerical_columns = ["writing_score","reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]

            # Build pipelines

            # handeling numerical values
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy = "median")),
                    ("scaler",StandardScaler())
                ]
            )

            # handeling categorical values
            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy = "most_frequent")),
                    ("one_hot_encoder",OneHotEncoder()),
                ]
            )

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")
            logging.info("Encoded categorical columns scaled successfully")
            
            # Combine numerical and categorical pipelines using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns), 
                    ("cat_pipeline",cat_pipeline,categorical_columns)
                ]
            )

            # Returns the configured transformer
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    # 2. USE transformer (execution)
    def initiate_data_transformation(self,train_path,test_path):
        try:
            # Load data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Read train and test data completed")
            
            logging.info("obtaining preprocessing object")
            
            # Like building a coffee machine without coffee beans
            # — refer to Example Working 1 below for context
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score","reading_score"]

            # Separate input features (X_train) and target variable (y_train) from the training dataset
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1) # X_train
            target_feature_train_df = train_df[target_column_name] # y_train

            # Separate input features (X_test) and target variable (y_test) from the test dataset
            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1) # X_test
            target_feature_test_df = test_df[target_column_name] # y_test

            logging.info("Applying preprocessing object on training dataframe and testing dataframe.")

            # When you call fit_transform, the data flows through the pipelines:
            # Like putting coffee beans in the machine and getting coffee
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # np.c_[] is a NumPy object for column-wise array concatenation. It stacks arrays horizontally (side by side as columns)
            # — refer to Example Working 2 below for context
            train_arr = np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            logging.info("Saved preprocessing object.")

            # Save the preprocessor object as a pickle file using the save_object function from utils.py
            save_object (
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e,sys)
