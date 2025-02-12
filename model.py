import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle  # Для сохранения модели в .pkl

# Загружаем датасет
data = pd.read_csv("heart.csv")

# Примерная подготовка данных
X = data.drop(columns=["target"])  # Фичи
y = data["target"]  # Целевая переменная

# Разделение данных на тренировочную и тестовую выборку
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Инициализация MLflow
mlflow.start_run()

# Логируем параметры
mlflow.log_param("model_type", "RandomForest")
mlflow.log_param("n_estimators", 100)

# Создаем модель
model = RandomForestClassifier(n_estimators=100)

# Обучаем модель
model.fit(X_train, y_train)

# Предсказания
y_pred = model.predict(X_test)

# Вычисление метрик
accuracy = accuracy_score(y_test, y_pred)

# Логируем метрики
mlflow.log_metric("accuracy", accuracy)

# Логируем модель в MLflow
mlflow.sklearn.log_model(model, "model")

# Сохраняем модель в формате .pkl
with open("random_forest_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Завершаем эксперимент MLflow
mlflow.end_run()

print(f"Model accuracy: {accuracy}")
