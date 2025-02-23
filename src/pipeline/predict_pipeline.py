import os,sys
from src.exception import CustomException
from src.utlis import load_obj
import pandas as pd

class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self,features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_obj(file_path=model_path)
            preprocessor = load_obj(file_path=preprocessor_path)
            data = preprocessor.transform(features)
            preds=model.predict(data)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)
    
class CustomData:
    def __init__(self,gender,race_ethnicity,parental_level_of_education,
                 lunch,test_preparation_course,
                 reading_score,writing_score):
        self.gender = gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
        
    def get_data_as_df (self):
        try:
            dict = {
                "gender":[self.gender],
                "race_ethnicity":[self.race_ethnicity],
                "parental_level_of_education":[self.parental_level_of_education],
                "lunch":[self.lunch],
                "test_preparation_course":[self.test_preparation_course],
                "reading_score":[self.reading_score],
                "writing_score":[self.writing_score]   
            }
            return pd.DataFrame(dict)
        
         
        except Exception as e:
            raise CustomException(e,sys)