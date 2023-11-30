
import pandas as pd
import numpy as np
from src.Income_prediction.logger import logging
from src.Income_prediction.exception import CustomException

import os
import sys
from sklearn.model_selection import train_test_split

from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    raw_data_path:str=os.path.join('artifacts',"raw.csv") 
    train_data_path:str=os.path.join("artifacts","train.csv")
    test_data_path:str=os.path.join("artifacts","test.csv")
    
class DatIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()
        
    def initiate_dataingestion(self):
        logging.info("Data ingestion started ")
        
        try:
            data=pd.read_csv(os.path.join("notebooks\data\credit_card_defaulter.csv"))
            logging.info("Read the dataset as data")
            
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_configraw_data_path)),exist_ok=True)
            data.to_csv(self.ingestion_config.raw_data_path,index=False) 
            logging.info("I have saved the raw data in artifact folder")
            
            logging.info("Here i am performing train test split")
            train_data,test_data=train_test_split(data,test_size=0.25)
            logging.info("Train test split completed")
            
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            
            logging.info("Data ingestion part completed")
            
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )   
            
        except Exception as e:
            logging.info("Exception occuring in dataingestion part")
            raise CustomException(e,sys)          
        