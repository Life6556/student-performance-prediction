import sys,os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from src.logger import logging

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,input_df):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join("artifacts","preprocessor.pkl")
            model=load_object(model_path)
            preprocessor=load_object(preprocessor_path)
            transformed_input_data=preprocessor.transform(input_df)
            preds=model.predict(transformed_input_data)
            logging.info("prediction successfull")
            return preds
        except Exception as e:
            logging.error(e)
            raise CustomException(e,sys)
            

class CustomData:
    def __init__(self,
                gender,
                race_ethnicity,
                parental_level_of_education,
                lunch,
                test_preparation_course,
                reading_score,
                writing_score):
        self.gender=gender
        self.race_ethnicity=race_ethnicity
        self.parental_level_of_education=parental_level_of_education
        self.lunch=lunch
        self.test_preparation_course=test_preparation_course
        self.reading_score=reading_score
        self.writing_score=writing_score
    def get_input_data_as_dataFrame(self):
        try:
            custom_data_input_dict = {
                            "gender": [self.gender],
                            "race_ethnicity": [self.race_ethnicity],
                            "parental_level_of_education": [self.parental_level_of_education],
                            "lunch": [self.lunch],
                            "test_preparation_course": [self.test_preparation_course],
                            "reading_score": [self.reading_score],
                            "writing_score": [self.writing_score],
                        }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            logging.exception("an error occured")
            raise CustomException(e,sys)

        
