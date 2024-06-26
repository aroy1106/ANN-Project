import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.components.data_transformation import DataTranformationConfig
from src.components.data_transformation import DataTransformation

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','raw_data.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiateDataIngestion(self):
        logging.info('Entered the Data Ingestion component.')

        try:
            df = pd.read_csv('data\Churn_Modelling.csv')

            logging.info('Successfully read CSV file.')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok = True)

            df.to_csv(self.ingestion_config.raw_data_path, index = False, header = True)

            logging.info('Train test split initiated.')

            train_set, test_set = train_test_split(df,test_size = 0.2)
            train_set.to_csv(self.ingestion_config.train_data_path, index = False, header = True)
            test_set.to_csv(self.ingestion_config.test_data_path, index = False, header = True)

            logging.info('Data Ingestion completed.')

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e, sys)
    
if __name__=='__main__':
    obj = DataIngestion()
    train_data, test_data = obj.initiateDataIngestion()

    data_transformation = DataTransformation()
    train_arr, test_arr, _ = data_transformation.initiateDataTransformation(train_data, test_data)
    print(train_arr.shape, test_arr.shape)