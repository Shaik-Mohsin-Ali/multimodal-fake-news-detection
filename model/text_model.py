import requests
import os

API_URL = "https://api-inference.huggingface.co/models/mrm8488/bert-tiny-finetuned-fake-news-detection"

HF_TOKEN = os.getenv("HF_TOKEN")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

def predict_text(text):

    payload = {"inputs": text[:256]}

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=10)
        result = response.json()

        # HuggingFace returns list of predictions
        label = result[0]["label"]
        score = result[0]["score"]

        return label, score

    except Exception as e:
        print("Text model error:", e)
        return "Unknown", 0.0