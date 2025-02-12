from sklearn.preprocessing import OneHotEncoder,StandardScaler
import os,sys
from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer 
from sklearn.compose import ColumnTransformer
import pandas as pd
import numpy as np
from src.utlis import save_obj
from dataclasses import dataclass
from src.exception import CustomException

@dataclass
class DataTransformationConfig :
    preprocessor_file_path = os.path.join("artifacts","preprocessor.pkl")
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
    
    def get_data_transformer_object(self):
        try:
            num_features = ['reading_score','writing_score']
            cat_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("onehotencoder",OneHotEncoder()),
                    ("scaler",StandardScaler(with_mean=False))
                ]
            )
            
            logging.info(f"Numerical columns : {num_features}")
            logging.info(f"Categorical columns :{cat_features}")
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,num_features),
                    ("cat_pipeline",cat_pipeline,cat_features)
                ]
            )
            
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
            
    
    def init_data_transformation(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info("Reading the train and test data")
            preprocessor_obj = self.get_data_transformer_object()
            target_col_name = "math_score"
            
            input_train_df = train_df.drop(columns=[target_col_name],axis=1)
            target_train_df = train_df[target_col_name]
            
            input_test_df = test_df.drop(columns=[target_col_name],axis=1)
            target_test_df = test_df[target_col_name]
            logging.info("Applying preprocessing on train and test set")
            
            input_feat_train_arr=preprocessor_obj.fit_transform(input_train_df)
            input_feat_test_arr=preprocessor_obj.transform(input_test_df)
            
            
            train_arr = np.c_[
                input_feat_train_arr,np.array(target_train_df)
            ]
            
            test_arr = np.c_[
                input_feat_test_arr,np.array(target_test_df)
            ]            
            
            logging.info("Saving preprocessor objects")
            
            save_obj(file_path=self.data_transformation_config.preprocessor_file_path,obj=preprocessor_obj)
            
            return(
                train_arr,test_arr,self.data_transformation_config.preprocessor_file_path
            )
        except Exception as e:
            raise CustomException(e,sys)

