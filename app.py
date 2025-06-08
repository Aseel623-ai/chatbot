from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
import google.generativeai as genai
import os
import io
import cv2
import numpy as np
from dotenv import load_dotenv
import json

# تحميل متغيرات البيئة
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("models/gemini-2.5-flash-preview-04-17")

app = FastAPI()

# تفعيل CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # عشان تتجرب محليًا بسهولة
    allow_methods=["*"],
    allow_headers=["*"],
)

# تعريف الموديل للدردشة
class ChatRequest(BaseModel):
    message: str

# دالة الشات بوت
@app.post("/chat")
async def chat_bot(request: ChatRequest):
    prompt = f"""
تم إدخال الأعراض التالية:
{request.message}

يرجى:

تقديم تقييم مبدئي بناءً على هذه الأعراض.
اقتراح التخصص الطبي المناسب.
إعطاء نصيحة عامة إن أمكن.
عدم تقديم علاج أو تشخيص نهائي.
"""
    try:
        response = model.generate_content(prompt)
        return {"reply": response.text.strip()}
    except Exception as e:
        return {"error": str(e)}


# تحسين الصورة باستخدام OpenCV
def enhance_image(contents: bytes) -> Image.Image:
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast = clahe.apply(denoised)
    _, binary = cv2.threshold(contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary)


# دالة تحليل التقرير الطبي المصور
@app.post("/analyze")
async def analyze_image(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        img = enhance_image(contents)
        extracted_text = pytesseract.image_to_string(img, lang='ara+eng')

        if not extracted_text.strip():
            return {"error": "لم يتم استخراج أي نص من الصورة. تأكد من وضوحها."}

        # إعداد الـ Prompt للنموذج
        prompt = f"""
تم استخراج تقرير طبي يحتوي على المعلومات التالية:
{extracted_text}

يرجى:
تلخيص التقرير بلغة بسيطة.
تحديد الحالة الأساسية.
عرض القيم غير الطبيعية.
تقديم 3-5 نصائح عملية.
تحديد الطبيب المناسب للمراجعة.
صيّغ الرد كمحتوى منظم بصيغة JSON.
"""

        response = model.generate_content(prompt)

        # نحاول نرجع JSON منسق إذا قدر يفهم
        try:
            organized = json.loads(response.text)
            return {
                "text_extracted": extracted_text.strip(),
                "analysis": organized
            }
        except:
            return {
                "text_extracted": extracted_text.strip(),
                "raw_diagnosis": response.text.strip(),
                "note": "لم يتمكن النموذج من تنسيق الرد كـ JSON تلقائياً. عرضنا الرد النصي بدلاً من ذلك."
            }

    except Exception as e:
        return {"error": f"حدث خطأ أثناء تحليل الصورة: {str(e)}"}
