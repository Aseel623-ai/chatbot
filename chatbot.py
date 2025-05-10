from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
import pytesseract
import io
import google.generativeai as genai

# إعداد Gemini API
genai.configure(api_key="YOUR_API_KEY")  # ← عدّلها بمفتاحك

# إعداد Tesseract OCR (غير ضروري في سيرفر Linux غالبًا)
pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

# إعداد FastAPI
app = FastAPI()

# السماح بالاتصال من أي مكان (للجبهة الأمامية مثلاً)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# نموذج للمحادثة
class ChatRequest(BaseModel):
    user_input: str

@app.post("/chat")
def chat(request: ChatRequest):
    user_input = request.user_input.strip()

    prompt = f"""أنت مساعد طبي ذكي تابع لإدارة طبية.
مهمتك:
- فهم الأعراض أو السؤال المقدم من المستخدم
- تقديم تشخيص مبدأي إن أمكن
- إعطاء نصيحة طبية مناسبة
- تحديد التخصص الطبي الذي يجب على المستخدم زيارته

السؤال أو الأعراض: {user_input}
"""

    try:
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
        response = model.generate_content([{"role": "user", "parts": [prompt]}])
        return {"response": response.text}
    except Exception as e:
        return {"error": str(e)}

@app.post("/upload")
async def upload_analysis(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        extracted_text = pytesseract.image_to_string(image, lang="ara+eng")

        prompt = f"""أنت مساعد طبي ذكي تابع للإدارة الطبية.
مهمتك:
- قراءة نتائج التحاليل المرفقة.
- تلخيص النتائج بشكل تقرير طبي واضح.
- تقديم نصيحة طبية عامة حسب النتائج.
- اقتراح التخصص الطبي المناسب إن لزم.

نتائج التحاليل المستخرجة:
{extracted_text}
"""
        model = genai.GenerativeModel(model_name="models/gemini-1.5-pro")
        response = model.generate_content([{"role": "user", "parts": [prompt]}])
        return {
            "extracted_text": extracted_text,
            "diagnosis": response.text
        }

    except Exception as e:
        return {"error": str(e)}
