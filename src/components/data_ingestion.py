import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts',"train.csv")
    test_data_path:str = os.path.join('artifacts',"test.csv")
    raw_data_path:str = os.path.join('artifacts',"data.csv")
    
    
class DataIngestion:
    def __init__(self):
        self.init_config = DataIngestionConfig()
    
    def init_data_ingestion(self):
        logging.info("Entered into the ingestion method")
        try:
            df = pd.read_csv('notebook\\data\\stud.csv')
            logging.info("Read the data set")
            os.makedirs(os.path.dirname(self.init_config.train_data_path),exist_ok=True)
            df.to_csv(self.init_config.raw_data_path,index=False,header=True)
            logging.info("Train test set splitting")
            train_set,test_set = train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.init_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.init_config.test_data_path,index=False,header=True)
            logging.info("Train test set splitted")

            
            
            return(
                self.init_config.train_data_path,
                self.init_config.test_data_path
            )
            
        except Exception as e:
            raise CustomException(e,sys)



if __name__ == "__main__":
    obj = DataIngestion()
    obj.init_data_ingestion()