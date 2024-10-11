import os
import sys
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.target_scaler = MinMaxScaler()
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["Open", "High", "Low", "Adj Close", "Volume"]

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            
            logging.info(f"Columns in the training dataframe: {train_df.columns}")
            logging.info(f"Columns in the testing dataframe: {test_df.columns}")

            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Close"
            numerical_columns = ["Open", "High", "Low", "Adj Close", "Volume"]

            numerical_columns = [col for col in numerical_columns if col in train_df.columns]
            if not numerical_columns:
                raise CustomException("No numerical columns found in the dataset", sys)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[[target_column_name]]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[[target_column_name]]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info("Scaling the target (Close) column")
            target_feature_train_arr = self.target_scaler.fit_transform(target_feature_train_df)
            target_feature_test_arr = self.target_scaler.transform(target_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Saving the preprocessor object")

            save_object(
            file_path=self.data_transformation_config.preprocessor_obj_file_path,
            obj=preprocessing_obj
        )

            return (
            train_arr,
            test_arr,
            self.data_transformation_config.preprocessor_obj_file_path,
        )

        except Exception as e:
            raise CustomException(e, sys)

        


        
