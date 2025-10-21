# ==========================================================
# Flask ML Example (Educational Version)
#
# الهدف: شرح بسيط لتشغيل موديل Machine Learning كـ API باستخدام Flask
# ==========================================================

from flask import Flask, request, jsonify
import joblib
import numpy as np

# ----------------------------------------------------------
# 1. إنشاء تطبيق Flask
# ----------------------------------------------------------
app = Flask(__name__)

# ----------------------------------------------------------
# 2. تحميل الموديل (تم حفظه مسبقًا بـ joblib)
# ----------------------------------------------------------
try:
    model = joblib.load("model.pkl")
except Exception as e:
    model = None
    print(f"[Warning] Model not loaded: {e}")

# ----------------------------------------------------------
# 3. تعريف الـ Routes (Endpoints)
# ----------------------------------------------------------

@app.route("/", methods=["GET"])
def home():
    """Endpoint رئيسي بسيط"""
    return jsonify({"message": "Welcome to Flask ML Example"})


@app.route("/users", methods=["GET"])
def get_users():
    """مثال بسيط لعرض بيانات"""
    return jsonify({"users": ["Ahmed", "Sara", "Omar"]})


@app.route("/users", methods=["POST"])
def create_user():
    """مثال توضيحي لإنشاء مستخدم"""
    data = request.get_json()
    name = data.get("name", "Unknown")
    return jsonify({"message": f"User '{name}' created successfully!"})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Endpoint للتنبؤ
    يستقبل features كـ JSON ويرجع نتيجة الموديل
    """
    if model is None:
        return jsonify({"error": "Model not loaded."}), 500

    data = request.get_json()
    features = data.get("features")

    if not features:
        return jsonify({"error": "Missing 'features' in request body."}), 400

    try:
        features_array = np.array([features])
        prediction = model.predict(features_array)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 400


# ----------------------------------------------------------
# 4. تشغيل التطبيق (Development mode)
# ----------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)

# ----------------------------------------------------------
# كيفية التشغيل:
# python flask_api_example.py
#
# يمكن اختبار الـ API باستخدام Postman أو curl.
# ----------------------------------------------------------
