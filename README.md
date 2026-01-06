# Medical Insurance Cost Prediction

## Overview
This project is an end-to-end machine learning application that predicts medical insurance charges based on demographic and lifestyle features.  
It demonstrates a complete ML workflow from data preprocessing to model deployment using Flask.

---

## Problem Statement
Medical insurance costs depend on factors such as age, BMI, smoking habits, and region.  
The objective of this project is to build a regression model that estimates insurance charges using these features.

---

## Dataset
The dataset contains the following columns:
- age  
- sex  
- bmi  
- children  
- smoker  
- region  
- charges (target variable)

---

## Project Structure
Medical-Insurance-Project/
│
├── src/mlProject/
│ ├── components/
│ │ ├── data_ingestion.py
│ │ ├── data_transformation.py
│ │ └── model_trainer.py
│ │
│ ├── pipeline/
│ │ ├── train_pipeline.py
│ │ └── predict_pipeline.py
│
├── notebooks/
│ └── EDA.ipynb
│
├── templates/
│ └── index.html
│
├── app.py
├── main.py
├── requirements.txt
└── README.md


---

## Workflow
1. Data ingestion and train-test split  
2. Feature engineering using scaling and one-hot encoding  
3. Model training and evaluation  
4. Saving trained model and preprocessing pipeline  
5. Flask-based web application for predictions  

---

## Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- Flask  
- Git & GitHub  


