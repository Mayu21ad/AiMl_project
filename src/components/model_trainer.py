import os
import sys
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, Dropout
from sklearn.metrics import r2_score
from tensorflow.keras.callbacks import EarlyStopping
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "cnn_lstm_model.h5")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            # Splitting the dataset to ensure the 'Close' column (index 4) is the target variable.
            X_train, y_train, X_test, y_test = (
                train_array[:, [0, 1, 2, 3, 5]],  # All columns except 'Close'
                train_array[:, 4],  # 'Close' as target variable
                test_array[:, [0, 1, 2, 3, 5]],
                test_array[:, 4],
            )

            # Reshape input data for CNN-LSTM
            X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
            X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

            # Building the CNN-LSTM model
            model = Sequential()

            model.add(Conv1D(filters=256, kernel_size=2, activation='relu', input_shape=(X_train.shape[1], 1)))
            model.add(MaxPooling1D(pool_size=4))
            model.add(LSTM(200, return_sequences=False))
            model.add(Flatten())
            model.add(Dense(200, activation='relu'))
            model.add(Dropout(0.3))
            model.add(Dense(1))

            model.compile(optimizer='Adam', loss='mean_squared_error')

            # Early stopping to prevent overfitting
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

            # Model training with validation and early stopping
            model.fit(X_train, y_train, epochs=200, batch_size=128, validation_data=(X_test, y_test), 
                      callbacks=[early_stopping], verbose=1)

            # Save the model
            model.save(self.model_trainer_config.trained_model_file_path)

            # Predictions
            predicted = model.predict(X_test)

            # Calculate R² score
            r2_square = r2_score(y_test, predicted)

            # Detailed process summary
            process_summary = {
                "Model Type": "CNN-LSTM Hybrid",
                "Training Epochs": 200,
                "Batch Size": 128,
                "Dropout": 0.3,
                "Filters in Conv1D": 256,
                "LSTM Units": 200,
                "R² Score": r2_square,
                "Input Shape": X_train.shape,
            }

            logging.info(f"Model training complete. R² Score: {r2_square}")

            # Return the model, process summary, and R² score
            return model, process_summary, r2_square

        except Exception as e:
            raise CustomException(e, sys)
