import requests
import os

API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def predict_text(text):

    payload = {"inputs": text[:256]}

    response = requests.post(API_URL, headers=headers, json=payload, timeout=10)

    data = response.json()

    # Handle API error
    if isinstance(data, dict) and "error" in data:
        print("HuggingFace API error:", data["error"])
        return "Unknown", 0.0

    # Normal prediction
    result = data[0]

    label = result["label"]
    score = result["score"]

    return label, score