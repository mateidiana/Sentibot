from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]


def detect_emotion(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    predicted_id = torch.argmax(probs).item()

    return {
        "emotion": LABELS[predicted_id],
        "confidence": float(probs[0][predicted_id])
    }
