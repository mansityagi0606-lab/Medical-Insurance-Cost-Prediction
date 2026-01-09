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

## Dockerized Deployment

This project is fully containerized using Docker, which allows the entire machine learning API (model + preprocessing + FastAPI server) to run in a reproducible and production-ready environment.

Using Docker ensures:

The same code runs identically on any machine

All dependencies are packaged together

The API can be deployed easily on cloud platforms such as AWS
Running the Project Using Docker

Step 1: Build the Docker image
From the project root directory, run:
docker build -t insurance-ml .

This creates a Docker image containing:
 The trained machine learning model
 The preprocessing pipeline
 The FastAPI application
 
Step 2: Run the Docker container
docker run -p 8000:8000 insurance-ml

Step 3: Access the API
http://localhost:8000/docs
