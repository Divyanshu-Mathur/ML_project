import pandas as pd
import numpy as np
import os,sys
import dill
from sklearn.metrics import r2_score

from src.exception import CustomException
from sklearn.model_selection import GridSearchCV


def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        
        with open(file_path,"wb") as file:
            dill.dump(obj,file)
            
    except Exception as e:
        raise CustomException(e,sys)
    


def evaluate_model(x_train,y_train,x_test,y_test,models,param):
    try:
        report={}
        for i in range(len(models)):
            model = list(models.values())[i]
            print(list(models.keys())[i])
            
            para = (list(param.values())[i])
            gs = GridSearchCV(model,param_grid=para,cv=3,n_jobs=-1)
            gs.fit(x_train,y_train)
            
            if hasattr(model, 'set_params'):
                model.set_params(**gs.best_params_)
            model.fit(x_train,y_train)
            
            
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)
            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)
            report [list(models.keys())[i]]=test_model_score
        return report
            
    except Exception as e :
        raise CustomException(e,sys)
    
    

def load_obj(file_path):
    try:
        with open(file_path,"rb") as file:
            return dill.load(file)
    except Exception as e:
        raise CustomException(e,sys)