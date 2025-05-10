from fastapi import FastAPI, UploadFile, File
import google.generativeai as genai
import pytesseract
from PIL import Image
import os

# إعداد مفتاح Gemini API (يفضل استخدام متغير بيئة في الإنتاج)
genai.configure(api_key="YOUR_API_KEY")

# تحديد مكان Tesseract (اختياري إذا كنت على لينكس وتم تركيبه بشكل صحيح)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# تهيئة النموذج
model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")

app = FastAPI()

@app.get("/")
def root():
    return {"message": "شات بوت طبي - FastAPI API يعمل بنجاح"}

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file)
        extracted_text = pytesseract.image_to_string(image, lang='ara+eng')

        messages = [
            {
                "role": "user",
                "parts": [f"""أنت مساعد طبي ذكي.
مهمتك هي:
- قراءة نتائج التحاليل الطبية.
- تلخيص النتائج في تقرير طبي واضح.
- تقديم نصيحة طبية عامة حسب النتائج.
- اقتراح التخصص الطبي المناسب.

نتائج التحاليل:
{extracted_text}"""]
            }
        ]

        response = model.generate_content(messages)
        return {"diagnosis": response.text}

    except Exception as e:
        return {"error": str(e)}
