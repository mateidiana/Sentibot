from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class Message(BaseModel):
    text: str


@router.post("/detect-emotion")
def detect_emotion(message: Message):
    # dummy response for now
    return {
        "emotion": "neutral",
        "confidence": 0.99,
        "input": message.text
    }
