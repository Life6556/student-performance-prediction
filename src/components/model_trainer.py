from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score

import os,sys
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.utils import evaluate_models,save_object
from src.exception import CustomException
from src.logger import logging

@dataclass
class ModelTrainerConfig:
    model_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_traing(self,train_arr,test_arr):
        X_train=train_arr[:,:-1]
        y_train=train_arr[:,-1]
        X_test=test_arr[:,:-1]
        y_test=test_arr[:,-1]

        models={
            "Linear Regression":LinearRegression(),
            "AdaBoost Regressor":AdaBoostRegressor(),
            "Random Forest":RandomForestRegressor(),
            "Gradient Boosting Regressor":GradientBoostingRegressor(),
            "K nearest neighbour":KNeighborsRegressor(),
            "SVR":SVR(),
            "XGBRegressor":XGBRegressor()
        }
        params={
                
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting Regressor":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "K nearest neighbour": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"]
                },
                "SVR":{
                    "kernel": ["linear", "rbf"],
                    "C": [0.1, 1, 10, 100],
                    
                }
                        
                }
        best_model,best_score=evaluate_models(X_train,y_train,X_test,y_test,models,params)
        
        save_object(
            file_path=self.model_trainer_config.model_path,
            obj=best_model
        )
        if best_score<0.6:
            raise CustomException("No best model found")
        logging.info("model found")
        return best_score


m=ModelTrainer()
dt=DataTransformation()
train_arr,test_arr=dt.initiateDataTranformation(r'artifacts\train.csv',r'artifacts\test.csv')
r_score=m.initiate_model_traing(train_arr,test_arr)
print(r_score)
        
