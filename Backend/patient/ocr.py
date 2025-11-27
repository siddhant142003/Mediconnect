import base64
import os
import requests
import re

groq_api_key = os.getenv("GROQ_API_KEY")

SYSTEM_PROMPT = """You are an advanced OCR tool. Your task is to accurately transcribe the text from the provided image.
Please follow these guidelines to ensure the transcription is as correct as possible:
1. Preserve Line Structure.
2. Avoid Splitting Words.
3. Correct Unnatural Spacing.
4. Recognize and Correct Word Breaks.
5. No Additional Comments.
6. Output text as it is, with correct spacing.
7. Do NOT summarize.
"""

def encode_image_to_base64(image):
    return base64.b64encode(image).decode("utf-8")

def parse_response(response_json):
    return response_json.get("choices", [{}])[0].get("message", {}).get("content", "")

def perform_ocr(image):
    base64_image = encode_image_to_base64(image)
    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {groq_api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "meta-llama/llama-4-maverick-17b-128e-instruct",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": SYSTEM_PROMPT},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }},
                    ],
                }
            ],
            "temperature": 0.2,
        },
    )

    if response.status_code != 200:
        return "OCR error"

    raw_text = parse_response(response.json())

    # ----------------------------------------------------
    # SAFE ABBREVIATION EXPANSION
    # ----------------------------------------------------
    abbrev_map = {
        r"\bOD\b": "once daily",
        r"\bBD\b": "twice daily",
        r"\bTID\b": "three times a day",
        r"\bQID\b": "four times a day",
        r"\bHS\b": "at bedtime",
        r"\bSOS\b": "if needed",
        r"\bPRN\b": "as needed",
        r"\bAC\b": "before meals",
        r"\bPC\b": "after meals",
        r"\bSTAT\b": "immediately",

        # Dosage schedules (strict detection)
        r"\b1\s*[-–—]\s*0\s*[-–—]\s*1\b": "one in the morning, none in the afternoon, one at night",
        r"\b1\s*[-–—]\s*1\s*[-–—]\s*1\b": "one in the morning, one in the afternoon, one at night",
        r"\b0\s*[-–—]\s*1\s*[-–—]\s*0\b": "one in the afternoon only",
        r"\b0\s*[-–—]\s*0\s*[-–—]\s*1\b": "one at night only",
        r"\b1\s*[-–—]\s*0\s*[-–—]\s*0\b": "one in the morning only",
        r"\b2\s*[-–—]\s*0\s*[-–—]\s*2\b": "two in the morning and two at night",
    }

    expanded_text = raw_text

    for pattern, meaning in abbrev_map.items():
        expanded_text = re.sub(pattern, lambda m: f"{m.group(0)} ({meaning})", expanded_text)

    return expanded_text
