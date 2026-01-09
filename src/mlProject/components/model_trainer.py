import os
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from mlProject.logger import logging
from mlProject.utils import save_object


class ModelTrainer:

    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model training with MLflow started")

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.01),
                "RandomForest": RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
            }

            mlflow.set_experiment("Medical_Insurance_Cost_Prediction")

            model_report = {}
            trained_models = {}

            for model_name, model in models.items():

                with mlflow.start_run(run_name=model_name):

                    model.fit(X_train, y_train)

                    y_train_pred = model.predict(X_train)
                    y_test_pred = model.predict(X_test)

                    train_r2 = r2_score(y_train, y_train_pred)
                    test_r2 = r2_score(y_test, y_test_pred)

                    # Log to MLflow
                    mlflow.log_param("model_name", model_name)
                    mlflow.log_metric("train_r2", train_r2)
                    mlflow.log_metric("test_r2", test_r2)
                    mlflow.sklearn.log_model(model, "model")

                    model_report[model_name] = test_r2
                    trained_models[model_name] = model

                    logging.info(f"{model_name} - Train R2: {train_r2}, Test R2: {test_r2}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = trained_models[best_model_name]

            mlflow.set_tag("best_model", best_model_name)

            logging.info(f"Best model found: {best_model_name}")

            save_object(self.model_path, best_model)

            return best_model_name, best_model_score

        except Exception as e:
            logging.error("Error in model training")
            raise e

