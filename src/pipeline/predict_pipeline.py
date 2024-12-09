import sys
import os
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
from tensorflow.keras.models import load_model
from src.components import model_trainer


class PredictPipeline:
    def __init__(self, model_type="cnn_lstm"):
        self.model_type = model_type

    def predict(self, features):
        try:
            if self.model_type == "cnn_lstm":
                model_path = os.path.join("artifacts", "cnn_lstm.h5")
                preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
                print("Before Loading CNN-LSTM Model")

                # Load the CNN-LSTM model
                model = load_object(file_path=model_path)
                
                # Load the preprocessor for scaling/transforming the features
                preprocessor = load_object(file_path=preprocessor_path)
                print("After Loading CNN-LSTM Model")

                # Preprocess the data
                data_scaled = preprocessor.transform(features)

                # Assuming the data needs to be reshaped like during training
                data_reshaped = data_scaled.reshape(data_scaled.shape[0], data_scaled.shape[1], 1)

                # Perform the prediction
                preds = model.predict(data_reshaped)

            else:
                raise ValueError("Unsupported model type")

            return preds

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    def __init__(self, feature_1, feature_2, feature_3, feature_4, feature_5):
        self.feature_1 = feature_1
        self.feature_2 = feature_2
        self.feature_3 = feature_3
        self.feature_4 = feature_4
        self.feature_5 = feature_5

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "feature_1": [self.feature_1],
                "feature_2": [self.feature_2],
                "feature_3": [self.feature_3],
                "feature_4": [self.feature_4],
                "feature_5": [self.feature_5],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)
