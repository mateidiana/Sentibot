import os
import logging
from google import genai  # ✅ updated import
from dotenv import load_dotenv

load_dotenv()  # make sure GEMINI_API_KEY is loaded


# Configure client
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

# Reference your model
MODEL_NAME = "models/gemini-3-pro-preview"

def get_gemini_suggestions(emotion: str, user_text: str) -> list[str]:
    prompt = f"""
You are a supportive assistant.

The user's emotional state is: {emotion}
User message: "{user_text}"

Give exactly 3 short, practical suggestions.
Use empathetic language.
Do NOT mention emotion detection or AI.
"""

    try:
        # Generate content from the model
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt,
            temperature=0.7,
            max_output_tokens=200
        )

        text = getattr(response, "text", "")
        if not text:
            logging.warning("Gemini returned empty suggestions.")
            return [
                "Try to take a deep breath.",
                "Step away for a moment.",
                "Talk to a friend."
            ]

        # Return list of suggestions
        return [line.strip() for line in text.split("\n") if line.strip()]

    except Exception as e:
        logging.error(f"Gemini API error: {e}")
        # fallback
        return [
            "Try to take a deep breath.",
            "Step away for a moment.",
            "Talk to a friend."
        ]
