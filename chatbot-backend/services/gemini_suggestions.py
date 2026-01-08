import os
import logging
from google import genai
from dotenv import load_dotenv
import re

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "models/gemini-3-pro-preview"

def get_gemini_suggestions(emotion: str, user_text: str) -> list[str]:
    prompt = f"""
You are a supportive assistant.

The user's emotional state is: {emotion}
User message: "{user_text}"

Give exactly 3 short, practical suggestions.
Use empathetic language.
Do NOT mention emotion detection or AI.
Provide each suggestion on a separate line.
"""

    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[prompt]
        )

        text = getattr(response, "text", "").strip()
        if not text:
            logging.warning("Gemini returned empty suggestions.")
            raise ValueError("Empty response")

        # Flexible parsing: remove numbering and empty lines
        lines = [re.sub(r'^\s*\d+\.?\s*', '', line).strip() for line in text.split("\n") if line.strip()]
        if len(lines) >= 3:
            return lines[:3]
        else:
            # fallback if Gemini returns fewer than 3 lines
            return [
                "Try to take a deep breath.",
                "Step away for a moment.",
                "Talk to a friend."
            ]

    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        return [
            "Try to take a deep breath.",
            "Step away for a moment.",
            "Talk to a friend."
        ]


if __name__ == "__main__":
    test_suggestions = get_gemini_suggestions("joy", "I just got a promotion at work!")
    print("Gemini suggestions:", test_suggestions)
