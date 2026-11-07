import os
import sys

import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logger import logging

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    

def evaluate_models(X_train,y_train,X_test,y_test,models,params):
    try:
        scores={}
        for name,model in models.items():
            model.fit(X_train,y_train)
            y_pred=model.predict(X_test)
            scores[name]=r2_score(y_test,y_pred)
        sorted_scores=sorted(scores.items(),key=lambda x:x[1],reverse=True)
        top_3models=dict(sorted_scores[:3])
        tuned_scores={}
        tuned_models={}
        for name in top_3models.keys():
            model=models[name]
            param=params.get(name,{})
            grid=GridSearchCV(estimator=model,param_grid=param,cv=3,scoring='r2',n_jobs=-1)
            grid.fit(X_train,y_train)
            best_model=grid.best_estimator_
            hp_tuned_ypred=best_model.predict(X_test)
            tuned_scores[name]=r2_score(y_test,hp_tuned_ypred)
            tuned_models[name]=best_model
        best_model_name=max(tuned_scores,key=tuned_scores.get)
        best_score=tuned_scores[best_model_name]
        best_model=tuned_models[best_model_name]

        return (best_model,best_score)
    except Exception as e:
        logging.exception("an error occurred")
        raise CustomException(e,sys)        
            
        
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
