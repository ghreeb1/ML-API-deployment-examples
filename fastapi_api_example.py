# ==========================================================
# FastAPI ML Example (Educational Version)
#
# الهدف: شرح بسيط لتشغيل موديل Machine Learning كـ API باستخدام FastAPI
# ==========================================================

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# ----------------------------------------------------------
# 1. تعريف تطبيق FastAPI
# ----------------------------------------------------------
app = FastAPI(
    title="FastAPI ML Example",
    description="Simple demo for serving a trained ML model as an API",
    version="1.0"
)

# ----------------------------------------------------------
# 2. تحميل الموديل (تم تدريبه مسبقًا وحُفِظ بـ joblib)
# ----------------------------------------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    model = None
    print(f"[Warning] Model not loaded: {e}")

# ----------------------------------------------------------
# 3. تعريف شكل البيانات المطلوبة من المستخدم (Input Schema)
# ----------------------------------------------------------
class InputData(BaseModel):
    features: list[float]  # قائمة أرقام تمثل القيم المدخلة للموديل


# ----------------------------------------------------------
# 4. Endpoints
# ----------------------------------------------------------

@app.get("/")
def home():
    """Endpoint رئيسي بسيط"""
    return {"message": "Welcome to FastAPI ML Example"}


@app.get("/users")
def get_users():
    """مثال بسيط لعرض بيانات"""
    return {"users": ["Ahmed", "Sara", "Omar"]}


@app.post("/users")
def create_user(name: str):
    """مثال توضيحي لإنشاء مستخدم"""
    return {"message": f"User '{name}' created successfully!"}


@app.post("/predict")
def predict(data: InputData):
    """
    Endpoint للتنبؤ
    يستقبل features كـ JSON ويرجع نتيجة الموديل
    """
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded.")

    try:
        features_array = np.array([data.features])
        prediction = model.predict(features_array)
        return {"prediction": prediction.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Prediction error: {e}")


# ----------------------------------------------------------
# 5. طريقة التشغيل (من PowerShell أو Terminal)
# ----------------------------------------------------------
# uvicorn fastapi_api_example:app --reload
#
# Swagger Docs:  http://127.0.0.1:8000/docs
# ReDoc:          http://127.0.0.1:8000/redoc
# ----------------------------------------------------------
