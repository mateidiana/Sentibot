from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# -----------------------
# Pre trained model (example)
# -----------------------

# MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"
# tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
# model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
# LABELS = ["anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise"]
#
#
# def detect_emotion(text: str):
#     inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
#
#     with torch.no_grad():
#         outputs = model(**inputs)
#
#     logits = outputs.logits
#     probs = torch.softmax(logits, dim=1)
#
#     predicted_id = torch.argmax(probs).item()
#
#     return {
#         "emotion": LABELS[predicted_id],
#         "confidence": float(probs[0][predicted_id])
#     }


# -----------------------
# Load trained model
# -----------------------

# Path to your trained model folder
MODEL_PATH = "../trained_emotion_model"

# Load tokenizer & model from the trained folder
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)

# Put model in evaluation mode for inference
model.eval()

# Labels used in your training
LABELS = ["sadness", "joy", "love", "anger", "fear", "surprise"]

# -----------------------
# Emotion detection function
# -----------------------


def detect_emotion(text: str):
    """
    Detects emotion from a single text input using the trained BERT classifier.
    Returns a dictionary with the predicted label and confidence score.
    """
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # Run model in inference mode
    with torch.no_grad():
        outputs = model(**inputs)

    # Get probabilities from logits
    logits = outputs.logits
    probs = torch.softmax(logits, dim=1)

    # Get predicted label index
    predicted_id = torch.argmax(probs).item()

    # Return emotion and confidence
    return {
        "emotion": LABELS[predicted_id],
        "confidence": float(probs[0][predicted_id])
    }
