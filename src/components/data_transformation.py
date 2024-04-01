import os
import sys
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src. utils import save_object


@dataclass
class DataTranformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTranformationConfig()

    def getDataTransformerObject(self):
        try:
            numerical_columns = ['Balance', 'EstimatedSalary']
            categorical_columns = ['Geography', 'Gender']

            num_pipeline = Pipeline(steps=[('scaler', StandardScaler())])
            logging.info('Numerical columns standard scaling completed.')

            cat_pipeline = Pipeline(steps=[('one_hot_encoder', OneHotEncoder())])
            logging.info('Categorical columns encoding completed.')

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline', num_pipeline, numerical_columns),
                    ('cat_pipeline', cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e, sys)
        
    def initiateDataTransformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info('Reading of train and test data completed.')
            logging.info('Obtaining preprocessing object.')

            preprocessing_obj = self.getDataTransformerObject()

            logging.info('Preprocessing object acquired.')

            target_column_name = "Exited"

            input_feature_train_df = train_df.drop(columns = [target_column_name], axis = 1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis = 1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and testing dataframe.')

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info('Saving preprocessing object.')

            save_object(
                file_path = self.data_transformation_config.preprocessor_obj_file_path,
                obj = preprocessing_obj
            )

            logging.info('Preprocessor object saved successfully.')

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )
        except Exception as e:
            raise CustomException(e, sys)