from fastapi import APIRouter
from pydantic import BaseModel
from models.emotion_model import detect_emotion
from services.gemini_suggestions import get_gemini_suggestions

router = APIRouter()


class Message(BaseModel):
    text: str


@router.post("/detect-emotion")
def detect_emotion_endpoint(message: Message):
    # 1️⃣ Detect emotion locally
    result = detect_emotion(message.text)
    emotion = result.get("emotion", "neutral")

    # 2️⃣ Generate suggestions from Gemini
    suggestions = get_gemini_suggestions(emotion, message.text)

    # 3️⃣ Return combined response
    return {
        "input": message.text,
        "emotion": emotion,
        "confidence": result.get("confidence", 1.0),
        "suggestions": suggestions
    }
