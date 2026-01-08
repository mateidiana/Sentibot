import os
import google.generativeai as genai  # correct import

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

model = genai.GenerativeModel("models/gemini-3-pro-preview")


def get_gemini_suggestions(emotion: str, user_text: str) -> str:
    prompt = f"""
You are a supportive assistant.

The user's emotional state is: {emotion}
User message: "{user_text}"

Give exactly 3 short, practical suggestions.
Use empathetic language.
Do NOT mention emotion detection or AI.
"""

    response = model.generate_content(prompt)

    return response.text.strip()
