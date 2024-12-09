import os # this library is used to interact with OS and file path manipulations
import sys # Provides access to specific system parameters and functions
import pandas as pd # for data manipulation and analysis
import numpy as np # for numerical operations and array manipulations

# Scikit-learn components for data preprocessing
from sklearn.preprocessing import MinMaxScaler # Scaling numerical features to the range of (0, 1)
from sklearn.compose import ColumnTransformer # for applying diff processing pipelines to specific columns
from sklearn.impute import SimpleImputer # for filling missing values in dataset
from sklearn.pipeline import Pipeline # creating pipelines for sequence preprocessing steps
from src.exception import CustomException # Custom exception class for handling errors
from src.logger import logging # Logger for capturing log information
from src.utils import save_object # Utility function to save objects to file

# Dataclass for configuration
from dataclasses import dataclass # a decorator for storing classes that store configuration data

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl") # path to save the preprocessor object

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig() # configuration details
        self.target_scaler = MinMaxScaler() # scaler for target variable
    
    def get_data_transformer_object(self):
        try:
            numerical_columns = ["Open", "High", "Low", "Adj Close", "Volume"] # columns that are numerical
            
            # Pipeline for the numerical columns for transformation
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", MinMaxScaler())
                ]
            )

            logging.info(f"Numerical columns: {numerical_columns}")
            
            # Combine preprocessing steps into a ColumnTransformer for selected columns
            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):     # Method to apply data transformation on training and testing datasets

        try:
            # Load training and testing datasets
            train_df = pd.read_csv(train_path) # Read training data from CSV file
            test_df = pd.read_csv(test_path)  # Read testing data from CSV file

            
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
            
            # Concatenate transformed features and target into final arrays for training and testing
            train_arr = np.c_[input_feature_train_arr, target_feature_train_arr]
            test_arr = np.c_[input_feature_test_arr, target_feature_test_arr]

            logging.info("Saving the preprocessor object")
            
            # Save the preprocessing object for future use
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

        


        
