from mlProject.components.data_ingestion import DataIngestion
from mlProject.components.data_transformation import DataTransformation
from mlProject.components.model_trainer import ModelTrainer
from mlProject.logger import logging


class TrainPipeline:
    def run_pipeline(self):
        logging.info("Training pipeline started")

        # 1. Data ingestion
        ingestion = DataIngestion()
        train_path, test_path = ingestion.initiate_data_ingestion()

        # 2. Data transformation
        transformation = DataTransformation()
        X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
            train_path, test_path
        )

        # 3. Model training
        trainer = ModelTrainer()
        trainer.initiate_model_trainer(X_train, y_train, X_test, y_test)

        logging.info("Training pipeline completed successfully")
