from mlProject.components.data_ingestion import DataIngestion
from mlProject.components.data_transformation import DataTransformation
from mlProject.components.model_trainer import ModelTrainer
from mlProject.pipeline.train_pipeline import TrainPipeline


if __name__ == "__main__":
    # Data Ingestion
    ingestion = DataIngestion()
    train_path, test_path = ingestion.initiate_data_ingestion()

    # Data Transformation
    transformation = DataTransformation()
    X_train, X_test, y_train, y_test = transformation.initiate_data_transformation(
        train_path, test_path
    )

    # Model Training
    trainer = ModelTrainer()
    best_model_name, best_model_score = trainer.initiate_model_trainer(
        X_train, y_train, X_test, y_test
    )

    print(f"Best Model: {best_model_name}")
    print(f"Best Model R2 Score: {best_model_score}")

    #train_pipeline
    pipeline = TrainPipeline()
    pipeline.run_pipeline()


