# Simple FastAPI App for Diabetes Prediction

from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle

# Load the best model (assumes model is saved as 'diabetes_model.pkl')
model_path = "app/diabetes_model.pkl"
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Initialize FastAPI
app = FastAPI(title="Diabetes Prediction API")

# Input schema
class DiabetesInput(BaseModel):
    Pregnancies: int
    Glucose: float
    BloodPressure: float
    SkinThickness: float
    Insulin: float
    BMI: float
    DiabetesPedigreeFunction: float
    Age: int

# Root endpoint
@app.get("/")
def read_root():
    return {"message": "Welcome to the Diabetes Prediction API"}

# Prediction endpoint
@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    input_data = np.array([
        data.Pregnancies,
        data.Glucose,
        data.BloodPressure,
        data.SkinThickness,
        data.Insulin,
        data.BMI,
        data.DiabetesPedigreeFunction,
        data.Age
    ]).reshape(1, -1)

    prediction = model.predict(input_data)[0]
    result = "Diabetic" if prediction == 1 else "Not Diabetic"
    return {"prediction": int(prediction), "result": result}
