# FastAPI & Flask ML Examples

## مشروع: FastAPI ML Example

هذا المشروع يوضّح مثالين مبسطين لتشغيل موديلات Machine Learning باستخدام كل من **FastAPI** و**Flask**.

---

## محتوى المشروع

* `fastapi_api_example.py` — خادم FastAPI يقوم بتحميل موديل وتقديم Endpoint للتنبؤ.
* `flask_api_example.py` — خادم Flask يقوم بنفس الوظيفة بطريقة أبسط.
* `model.pkl` — ملف الموديل المدرّب (يجب وضعه في نفس المجلد).

---

## المتطلبات

أنشئ ملف `requirements.txt` يحتوي على السطور التالية:

```
fastapi
uvicorn
flask
joblib
numpy
pydantic
```

تثبيت المتطلبات:

```bash
pip install -r requirements.txt
```

---

## تشغيل السيرفرات (محلياً)

### FastAPI

```bash
uvicorn fastapi_api_example:app --reload
```

* واجهة التوثيق (Swagger): [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

### Flask

```bash
python flask_api_example.py
```

* رابط التشغيل الافتراضي: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## مثال على طلب التنبؤ (Prediction Request)

Body بصيغة JSON:

```json
{
  "features": [5.1, 3.5, 1.4, 0.2]
}
```

---

## ملاحظات مهمة

* تحقّق من عدد القيم داخل المصفوفة (مثلاً: نموذج Iris يحتاج 4 قيم فقط).
* يفضّل تحميل الموديل داخل event مخصص (startup) في FastAPI بدلاً من التحميل أثناء الاستيراد.
* استخدم `try/except` حول `model.predict` لإرجاع أخطاء واضحة في حالة فشل التنبؤ.
* في Flask، يُفضل عدم تحميل الموديل عند الاستيراد مباشرة. يمكن نقله إلى دالة يتم استدعاؤها عند التشغيل.
* في بيئة الإنتاج: استخدم أدوات مثل `gunicorn` أو `uvicorn` مع أكثر من عامل (worker)، وأضف تسجيل (logging)، وفحص الحالة (health check)، والمصادقة (authentication) إذا لزم.

---

## تطوير إضافي (اختياري)

يمكنك إضافة ما يلي لتحسين المشروع:

* ملف `Dockerfile` لتسهيل النشر.
* آلية تحقق من أنواع البيانات في `pydantic`.
* فصل إعدادات البيئة في ملف `.env`.

---

ملحوظة: هذا المشروع تعليمي يهدف لتوضيح الفكرة وليس معدًّا للإنتاج بشكل مباشر.

## 📧 Contact

**Developer:**  
<h2 align="center">Mohamed Khaled</h2>

<p align="center">
  <a href="mailto:qq11gharipqq11@gmail.com" target="_blank">
    <img src="https://img.shields.io/badge/-Gmail-D14836?style=for-the-badge&logo=gmail&logoColor=white"/>
  </a>
  <a href="https://www.linkedin.com/in/mohamed-khaled-3a9021263" target="_blank">
    <img src="https://img.shields.io/badge/-LinkedIn-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white"/>
  </a>
</p>
