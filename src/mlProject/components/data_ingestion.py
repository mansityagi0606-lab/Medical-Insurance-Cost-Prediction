import os
import pandas as pd
from sklearn.model_selection import train_test_split
from mlProject.logger import logging

class DataIngestion:
    def __init__(self):
        self.artifact_dir = "artifacts"
        self.train_path = os.path.join(self.artifact_dir, "train.csv")
        self.test_path = os.path.join(self.artifact_dir, "test.csv")

    def initiate_data_ingestion(self):
        try:
            logging.info("Starting data ingestion")

            # Read raw data
            df = pd.read_csv("data/insurance.csv")

            # Create artifacts directory
            os.makedirs(self.artifact_dir, exist_ok=True)

            # Train-test split
            train_df, test_df = train_test_split(
                df, test_size=0.2, random_state=42
            )

            # Save files
            train_df.to_csv(self.train_path, index=False)
            test_df.to_csv(self.test_path, index=False)

            logging.info("Data ingestion completed")

            return self.train_path, self.test_path

        except Exception as e:
            raise e
