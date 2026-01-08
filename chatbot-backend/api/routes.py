from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from models.emotion_model import detect_emotion
from services.gemini_suggestions import get_gemini_suggestions

router = APIRouter()

class Message(BaseModel):
    text: str

@router.post("/detect-emotion")
def detect_emotion_endpoint(message: Message):
    try:
        # 1️⃣ Detect emotion locally
        result = detect_emotion(message.text)
        emotion = result.get("emotion", "neutral")
        confidence = result.get("confidence", 1.0)

        # 2️⃣ Get suggestions safely from Gemini
        suggestions = get_gemini_suggestions(emotion, message.text)

        # 3️⃣ Return combined response
        return {
            "input": message.text,
            "emotion": emotion,
            "confidence": confidence,
            "suggestions": suggestions
        }

    except Exception as e:
        # catch any unexpected errors
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")
