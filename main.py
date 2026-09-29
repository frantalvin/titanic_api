from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from pathlib import Path


app = FastAPI()


# Charger le modèle
model_path = Path(__file__).resolve().with_name("model.pkl")
if not model_path.is_file():
    raise FileNotFoundError(
        f"Modèle introuvable : {model_path}. Exécutez train_model.py pour le générer."
    )
model = joblib.load(model_path)


# Données reçues
class Passenger(BaseModel):
    Pclass: int
    Sex: str
    Age: float
    SibSp: int
    Parch: int
    Fare: float


@app.get("/")
def home():
    return {
        "message": "API Titanic opérationnelle"
    }


@app.post("/predict")
def predict(passenger: Passenger):

    data = pd.DataFrame([passenger.model_dump()])

    prediction = model.predict(data)

    return {
        "prediction": int(prediction[0])
    }