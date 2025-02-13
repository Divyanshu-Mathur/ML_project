import numpy as np
import pandas as pd
import os,sys
from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor,GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error,root_mean_squared_error,mean_absolute_error
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from src.utlis import evaluate_model,save_obj

@dataclass
class Model_train_config:
    train_model_file_path = os.path.join("artifacts","model.pkl")
    
    
class Model_train:
    def __init__(self):
        self.model_trainer_config = Model_train_config()
        
    def init_model_train(self,train_arr,test_arr):
        try:
            logging.info("Splitting train and test input data")
            x_train,y_train,x_test,y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]
            )
            
            models = {
                "Linear Regression":LinearRegression(),
                "Ridge":Ridge(),
                "Lasso":Lasso(),
                "Elastic Net":ElasticNet(),
                "XGB":XGBRegressor(),
                "KNN":KNeighborsRegressor(),
                "DT":DecisionTreeRegressor(),
                "Random Forest":RandomForestRegressor(),
                "AdaBoost":AdaBoostRegressor(),
                "Gradient":GradientBoostingRegressor()
            }
            
            params={
                "Linear Regression":{},
                "Ridge":{},
                "Lasso":{},
                "Elastic Net":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                "KNN":{
                    'n_neighbors':[i for i in range(1,10)],
                    'weights':['uniform', 'distance'],
                    'algorithm':['auto', 'ball_tree', 'kd_tree', 'brute']  
                },
                
                "DT": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    'splitter':['best','random'],
                    'max_features':['sqrt','log2']
                },
                "Random Forest":{
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                
                
                
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.7, 0.8, 0.9],
                    'criterion':['squared_error', 'friedman_mse'],
                    'max_features':['auto','sqrt','log2'],
                    'n_estimators': [32,64,128]
                }
                
            }
            
            model_report:dict = evaluate_model(x_train=x_train,y_train=y_train,models=models,x_test=x_test,y_test=y_test,param=params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            if best_model_score<0.6:
                raise CustomException("No best model")

            logging.info(f"Best model found and its name is {best_model_name}")
            save_obj(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
            
            return best_model_score
            
            
            
        except Exception as e:
            raise CustomException(e,sys)

        
        