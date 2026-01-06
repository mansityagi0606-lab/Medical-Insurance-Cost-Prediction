import os
import sys
import numpy as np

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from mlProject.logger import logging
from mlProject.utils import save_object


class ModelTrainer:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")

    def evaluate_model(self, X_train, y_train, X_test, y_test, model):
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        train_r2 = r2_score(y_train, y_train_pred)
        test_r2 = r2_score(y_test, y_test_pred)

        return train_r2, test_r2

    def initiate_model_trainer(self, X_train, y_train, X_test, y_test):
        try:
            logging.info("Model training started")

            models = {
                "LinearRegression": LinearRegression(),
                "Ridge": Ridge(alpha=1.0),
                "Lasso": Lasso(alpha=0.01),
                "RandomForest": RandomForestRegressor(
                    n_estimators=100,
                    random_state=42
                )
            }

            model_report = {}

            for model_name, model in models.items():
                train_r2, test_r2 = self.evaluate_model(
                    X_train, y_train, X_test, y_test, model
                )

                model_report[model_name] = test_r2
                logging.info(f"{model_name} - Train R2: {train_r2}, Test R2: {test_r2}")

            best_model_name = max(model_report, key=model_report.get)
            best_model_score = model_report[best_model_name]
            best_model = models[best_model_name]

            logging.info(f"Best model found: {best_model_name}")

            best_model.fit(X_train, y_train)
            save_object(self.model_path, best_model)

            return best_model_name, best_model_score

        except Exception as e:
            logging.error("Error in model training")
            raise e
