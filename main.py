import pickle
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd

app = FastAPI(title = "Prediction API")

#Load the trained model
with open("model.pkl", "wb") as f:
    model = pickle.load(f)


class Person(BaseModel):
    age: int
    sex: int
    oldpeak: float
    

@app.get("/")
def road_root():
    return {"message": "Welcome to Prediction Model API"}


@app.get("/")
def read_root():
    return {"message": "Welcome to Titanic Survival Prediction API"}


@app.post("/predict")
def predict(person: Person):
    data = person.dict()
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    return {"Survived": int(prediction)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)