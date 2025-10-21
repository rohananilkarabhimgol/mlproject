import pandas as pd
import sys
import os
from src.logger import logging # Custom logging module for standardized log messages across the project
from src.exception import CustomException # Custom exception class to handle and raise project-specific errors

from sklearn.model_selection import train_test_split # Utility to split dataset into training and testing subsets
from dataclasses import dataclass # Decorator to simplify class creation by automatically generating init and repr methods

from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig

# Clean - all paths in one place
@dataclass
class DataIngestionConfig:
    # Path: "artifacts/train.csv"
    train_data_path: str = os.path.join("artifacts","train.csv")
    test_data_path: str = os.path.join("artifacts","test.csv")
    raw_data_path: str = os.path.join("artifacts","data.csv")

class DataIngestion:
    def __init__(self):
        # 1. Setup paths
        self.ingestion_config = DataIngestionConfig()
    
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # 2. Read data
            df = pd.read_csv("notebook/data/stud.csv")
            logging.info("Reading the dataset into a DataFrame.")
            
            # 3. Create directory ONLY when we have data to save
            # Path: "artifacts/train.csv"
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True) # → "artifacts"
            # os.makedirs("artifacts", exist_ok=True)  # Creates "artifacts" folder
            
            # 4. Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # 5. Split and save train/test
            logging.info("Train-test split initiated.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            return (
                self.ingestion_config.train_data_path, 
                self.ingestion_config.test_data_path
                )
            
        except Exception as e:
            raise CustomException(e,sys)

# TEST BLOCK - for development only      
if __name__ == "__main__":
    # This is just to test if ingestion works
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Optional: Also test transformation
    data_transformation = DataTransformation()
    train_arr,test_arr,_ = data_transformation.initiate_data_transformation(train_data,test_data)

    modeltrainer = ModelTrainer()
    print(modeltrainer.initiate_model_trainer(train_arr,test_arr))