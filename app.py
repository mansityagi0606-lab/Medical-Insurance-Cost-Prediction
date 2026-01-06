from flask import Flask, request, render_template
import pandas as pd
from mlProject.pipeline.predict_pipeline import PredictPipeline

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "age": [int(request.form["age"])],
        "sex": [request.form["sex"]],
        "bmi": [float(request.form["bmi"])],
        "children": [int(request.form["children"])],
        "smoker": [request.form["smoker"]],
        "region": [request.form["region"]]
    }

    df = pd.DataFrame(data)
    pipeline = PredictPipeline()
    prediction = pipeline.predict(df)

    return render_template("index.html", result=f"Predicted Insurance Cost: ₹{prediction[0]:,.2f}")

if __name__ == "__main__":
    app.run(debug=True)
