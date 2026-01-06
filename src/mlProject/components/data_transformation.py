import sys
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from mlProject.utils import save_object
from mlProject.logger import logging

class DataTransformation:
    def __init__(self):
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["age", "bmi", "children"]
            categorical_columns = ["sex", "smoker", "region"]

            num_pipeline = Pipeline(
                steps=[
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("onehot", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise e

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            target_column = "charges"

            X_train = train_df.drop(columns=[target_column])
            y_train = np.log1p(train_df[target_column])

            X_test = test_df.drop(columns=[target_column])
            y_test = np.log1p(test_df[target_column])

            preprocessor = self.get_data_transformer_object()

            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            save_object(self.preprocessor_path, preprocessor)

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test
            )

        except Exception as e:
            raise e
