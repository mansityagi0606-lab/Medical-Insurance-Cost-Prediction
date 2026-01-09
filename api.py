from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

from mlProject.pipeline.predict_pipeline import PredictPipeline

app = FastAPI()

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

class InsuranceInput(BaseModel):
    age: int
    sex: str
    bmi: float
    children: int
    smoker: str
    region: str

@app.post("/predict")
def predict(data: InsuranceInput):
    df = pd.DataFrame([data.dict()])
    pipeline = PredictPipeline()
    result = pipeline.predict(df)

    return {"predicted_charges": float(result[0])}

@app.get("/health")
def health():
    return {"status": "running"}
