import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass

from src.Income_prediction.logger import logging
from src.Income_prediction.exception import CustomException

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from src.Income_prediction.utils.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()
        
    def get_data_transformation(self):
        try:
            logging.info("Data transformation initiated")
            cat_columns=["default","student"]
            num_columns=["balance"]
            
            logging.info("Pipline initiated")
            
            num_pipeline=Pipeline(
                steps=[("imputer",SimpleImputer(strategy="median")),
                       ('scaler',MinMaxScaler())]
            )
            
            cat_pipeline=Pipeline(
                steps=[('imputer',SimpleImputer(strategy="most_frequent")),
                       ('encoder',LabelEncoder()),
                       ('scaler',MinMaxScaler())]
            )
            
            preprocessor=ColumnTransformer([
                ('num_pipeline',num_pipeline,num_columns),
                ('cat_pipeline',cat_pipeline,cat_columns)
            ]
            )
            
            return preprocessor
        
        except Exception as e:
            logging.info("Error occured in get_data transformation")
            raise CustomException(e,sys)
        
        
    def initialize_dataTransformation(self,train_path,test_path):
        try:
            logging.info("Initialize Datatransformation")
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("train and test df stored")
            logging.info(f"Train_df head :\n{train_df.head().to_string()}")
            logging.info(f"Test_df head :\n{test_df.head().to_string()}")
            
            preprocessor_obj=self.get_data_transformation()
            
            target_column='income'
            drop_column=['id',target_column] 
            
            input_feature_train_df=train_df.drop(columns=drop_column,axis=1)
            target_feature_train_df=train_df[target_column]    
            
            input_feature_test_df=test_df.drop(columns=drop_column,axis=1)
            target_feature_test_df=test_df[target_column] 
            
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.fit_transform(input_feature_test_df)
            
            logging.info("Applying preprocessor obj on train and test df")
            
            train_arr=np.c_(input_feature_train_arr,np.array(target_feature_train_df))
            test_arr=np.c_(input_feature_test_arr,np.array(target_feature_test_df))
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )
            
            logging.info("Processor.pkl file saved")
            
            return (train_arr,test_arr)
            
        except Exception as e:
            logging.info("Error occured in ")
            raise CustomException(e,sys)
        
    

