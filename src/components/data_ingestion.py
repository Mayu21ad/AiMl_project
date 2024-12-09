# import os
# import sys
# from src.exception import CustomException
# from src.logger import logging
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from dataclasses import dataclass
# import yfinance as yf

# from src.components.data_transformation import DataTransformation
# from src.components.data_transformation import DataTransformationConfig
# from src.components.model_trainer import ModelTrainerConfig
# from src.components.model_trainer import ModelTrainer

# @dataclass
# class DataIngestionConfig:
#     train_data_path: str = os.path.join('artifacts', "train.csv")
#     test_data_path: str = os.path.join('artifacts', "test.csv")
#     raw_data_path: str = os.path.join('artifacts', "data.csv")

# class DataIngestion:
#     def __init__(self, ticker: str, start_date: str, end_date: str):
#         self.ingestion_config = DataIngestionConfig()
#         self.ticker = ticker
#         self.start_date = start_date
#         self.end_date = end_date

#     def initiate_data_ingestion(self):
#         logging.info("Entered the data ingestion method")
#         try:
#             logging.info(f"Downloading data for ticker: {self.ticker} from {self.start_date} to {self.end_date}")
#             df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
#             logging.info("The dataset is ready to download for the provided ticker")

#             os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

#             df.to_csv(self.ingestion_config.raw_data_path, index=True, header=True)

#             logging.info("Train Test split of data is initiated")

#             train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

#             train_set.to_csv(self.ingestion_config.train_data_path, index=True, header=True)
#             test_set.to_csv(self.ingestion_config.test_data_path, index=True, header=True)

#             logging.info("Data ingestion is done")

#             return(
#                 self.ingestion_config.train_data_path,
#                 self.ingestion_config.test_data_path
#             )
#         except Exception as e:
#             raise CustomException(e, sys)

# if __name__ == "__main__":
#     ticker = input("Enter the ticker symbol: ")
#     start_date = input("Enter the start_date (YYYY-MM-DD): ")
#     end_date = input("Enter the end_date (YYYY-MM-DD): ")

#     obj = DataIngestion(ticker=ticker, start_date=start_date, end_date=end_date)
#     train_data, test_data = obj.initiate_data_ingestion()

#     data_transformation = DataTransformation()
#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

#     modeltrainer = ModelTrainer()
#     print(modeltrainer.initiate_model_trainer(train_arr, test_arr))

import os
import sys
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
import yfinance as yf

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', 'train.csv')
    test_data_path: str = os.path.join('artifacts', 'test.csv')
    raw_data_path: str = os.path.join('artifacts', 'raw.csv')

class DataIngestion:
    def __init__(self, ticker, start_date, end_date):
        self.ingestion_config = DataIngestionConfig()
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion starts...")
        try:
            logging.info(f"Data ingestion for {self.ticker} from {self.start_date} to {self.end_date}")
            df = yf.download(self.ticker, start=self.start_date, end=self.end_date)

            # Check if the DataFrame is empty
            if df.empty:
                raise ValueError("No data received for the specified ticker and date range.")
            
            logging.info("Required stock data downloaded...")

            # Create directories if they do not exist
            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path), exist_ok=True)

            # Save raw data to CSV
            df.to_csv(self.ingestion_config.raw_data_path, index=True)
            logging.info(f"Raw data saved to {self.ingestion_config.raw_data_path}")

            # Split data into train and test sets
            logging.info("Splitting data into training and testing sets.")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            # Save train and test sets to CSV
            train_set.to_csv(self.ingestion_config.train_data_path, index=True)
            logging.info(f"Train data is saved to {self.ingestion_config.train_data_path}")

            test_set.to_csv(self.ingestion_config.test_data_path, index=True)
            logging.info(f"Test data is saved to {self.ingestion_config.test_data_path}")

            logging.info("Data Ingestion completed successfully")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        
        except Exception as e:
            logging.error(f"Error during data ingestion: {e}")
            raise

if __name__ == "__main__":
    # Prompt the user for input values.
    ticker = input("Enter the ticker symbol (e.g., AAPL, MSFT): ").strip().upper()  # Stock ticker symbol.
    start_date = input("Enter the start date (YYYY-MM-DD): ").strip()  # Start date for data.
    end_date = input("Enter the end date (YYYY-MM-DD): ").strip()  # End date for data.

    # Create an instance of the DataIngestion class with the provided inputs.
    data_ingestion = DataIngestion(ticker=ticker, start_date=start_date, end_date=end_date)
    try:
        # Start the data ingestion process and print the output file paths.
        train_data_path, test_data_path = data_ingestion.initiate_data_ingestion()
        print(f"Data ingestion completed.\nTrain data saved at: {train_data_path}\nTest data saved at: {test_data_path}")
    except Exception as e:
        # If any error occurs, display it to the user.
        print(f"Failed to complete data ingestion: {e}")
