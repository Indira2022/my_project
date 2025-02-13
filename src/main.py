from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np

# Создаем FastAPI приложение
app = FastAPI()

# Загружаем модель, сохраненную в формате .pkl
with open("random_forest_model.pkl", "rb") as f:
    model = pickle.load(f)

# Определяем модель данных для входных данных из датасета heart.csv
class InputData(BaseModel):
    age: float           # Возраст
    sex: int             # Пол (0 или 1)
    cp: int              # Тип боли в груди
    trestbps: float      # Кровяное давление в покое
    chol: float          # Холестерин
    fbs: int             # Уровень сахара в крови (0 или 1)
    restecg: int         # Электрокардиографический результат
    thalach: float       # Максимальная частота пульса
    exang: int           # Стенокардия (0 или 1)
    oldpeak: float       # Старое изменение депрессии
    slope: int           # Наклон пика упражнений
    ca: int              # Количество крупных сосудов
    thal: int            # Тип талассемии

# Эндпоинт для получения предсказания
@app.post("/predict/")
def predict(data: InputData):
    # Преобразуем данные в DataFrame
    input_data = pd.DataFrame([data.dict()])  # вызываем dict() на экземпляре объекта

    # Получаем предсказание от модели
    prediction = model.predict(input_data)

    return {"prediction": int(prediction[0])}
