import os
import numpy as np
import pandas as pd
import pickle

class PredictPipeline:
    def __init__(self):
        self.model_path = os.path.join("artifacts", "model.pkl")
        self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")

    def predict(self, input_data: pd.DataFrame):
        with open(self.preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)

        with open(self.model_path, "rb") as f:
            model = pickle.load(f)

        transformed_data = preprocessor.transform(input_data)
        prediction = model.predict(transformed_data)

        return np.expm1(prediction)  # reverse log1p
