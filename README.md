# Medical Insurance Cost Prediction

This project builds a complete machine learning system to predict medical insurance costs based on demographic and lifestyle information. It follows a real-world ML workflow, starting from raw data and ending with a deployed prediction API.

The objective is to estimate how much insurance a person is likely to be charged based on attributes such as age, BMI, smoking habits, and region.

---

## Problem Statement

Health insurance costs vary widely from person to person. Insurance companies use data-driven models to estimate expected medical expenses before deciding premiums.  
This project trains a regression model to predict medical insurance charges using historical data.

---

## Dataset

The dataset contains the following columns:

- age – Age of the person  
- sex – Gender  
- bmi – Body Mass Index  
- children – Number of dependents  
- smoker – Whether the person smokes  
- region – Residential region  
- charges – Medical insurance cost (target variable)

---

## Machine Learning Workflow

The project is built as a production-style ML pipeline:

1. **Data Ingestion**  
   Raw data is loaded and split into training and test sets.

2. **Data Preprocessing**  
   - Numerical features are scaled using StandardScaler  
   - Categorical features are encoded using OneHotEncoder  
   - Log transformation is applied to the target variable to reduce skewness  

3. **Model Training and Selection**  
   Multiple regression models are trained and evaluated.  
   The best performing model is selected automatically based on R² score.

4. **Model Saving**  
   The trained model and preprocessing pipeline are saved so they can be reused for prediction.

5. **Prediction Pipeline**  
   A separate pipeline loads the trained artifacts and generates predictions for new inputs.

---

## Model Performance

The final trained model achieves:

- R² score of approximately **0.85** on the test dataset  

This indicates good predictive accuracy for estimating insurance charges.

---

## Running the API

The trained model is served using FastAPI.

Start the server:

uvicorn app:app --reload
Open in browser:
http://127.0.0.1:8000

API documentation:
http://127.0.0.1:8000/docs
Send a POST request to `/predict`:
{
"age": 28,
"sex": "female",
"bmi": 24.5,
"children": 1,
"smoker": "no",
"region": "northwest"
}
The API returns the predicted insurance cost.

## Project Structure
src/mlProject/
│
├── components/
│ ├── data_ingestion.py
│ ├── data_transformation.py
│ ├── model_trainer.py
│ └── model_evaluation.py
│
├── pipeline/
│ ├── train_pipeline.py
│ └── predict_pipeline.py
│
├── utils.py
├── logger.py


The code is organized into modular components so each stage of the ML pipeline can be maintained and extended easily.

---

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- FastAPI  
- Uvicorn  
- Git and GitHub  

---

## Why this project

This project demonstrates how machine learning systems are built in practice — not just in notebooks.  
It shows data preprocessing, model training, evaluation, and deployment as a working API, making it suitable for real-world use.
