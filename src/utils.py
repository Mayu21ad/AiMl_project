# import os
# import sys
# import dill
# import pickle
# import numpy as np
# import pandas as pd
# from sklearn.metrics import r2_score
# from sklearn.model_selection import GridSearchCV

# from src.exception import CustomException

# def save_object(file_path, obj):
#     try:
#         dir_path = os.path.dirname(file_path)
#         os.makedirs(dir_path, exist_ok=True)
#         with open(file_path, 'wb') as file_obj:
#             pickle.dump(obj, file_obj)

#     except Exception as e:
#         raise CustomException(e, sys)
    

    
# def evaluate_models(X_train, y_train, X_test, y_test, model):
#     try:
#         model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
#         y_train_pred = model.predict(X_train)
#         y_test_pred = model.predict(X_test)
#         train_model_score = r2_score(y_train, y_train_pred)
#         test_model_score = r2_score(y_test, y_test_pred)

#         report = {
#             "Train R2 Score": train_model_score,
#             "Test R2 Score": test_model_score,
#         }
#         return report
#     except Exception as e:
#         raise CustomException(e, sys)

# def load_object(file_path):
#     try:
#         with open(file_path, 'rb') as file_obj:
#             return pickle.load(file_obj)
#     except Exception as e:
#         raise CustomException(e, sys)

import os
import sys
import pickle
from sklearn.metrics import r2_score
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise CustomException(e, sys)

def evaluate_model(X_train, y_train, X_test, y_test, model):
    try:
        # Train the model
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)
        
        # Predict and calculate R² score
        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_model_score = r2_score(y_train, y_train_pred)
        test_model_score = r2_score(y_test, y_test_pred)

        report = {
            "Train R² Score": train_model_score,
            "Test R² Score": test_model_score,
        }
        return report
    except Exception as e:
        raise CustomException(e, sys)

def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)

