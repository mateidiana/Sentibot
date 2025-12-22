from fastapi import APIRouter
from pydantic import BaseModel
from models.emotion_model import detect_emotion

router = APIRouter()


class Message(BaseModel):
    text: str


@router.post("/detect-emotion")
def detect_emotion_endpoint(message: Message):
    # dummy response for now
    # return {
    #     "emotion": "neutral",
    #     "confidence": 0.99,
    #     "input": message.text
    # }
    result = detect_emotion(message.text)
    return result
