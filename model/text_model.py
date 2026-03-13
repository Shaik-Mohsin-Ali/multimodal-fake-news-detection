import requests

API_URL = "https://api-inference.huggingface.co/models/distilbert-base-uncased-finetuned-sst-2-english"

def predict_text(text):

    payload = {"inputs": text[:256]}

    response = requests.post(API_URL, json=payload)

    result = response.json()[0]

    label = result["label"]
    score = result["score"]

    if label == "NEGATIVE":
        final_label = "Fake"
    else:
        final_label = "Real"

    return final_label, score


def explain_text(text):

    suspicious_words = [
        "breaking","shocking","secret","unbelievable",
        "aliens","ufo","miracle","hoax","conspiracy"
    ]

    words = text.split()

    highlighted = []
    found = []

    for w in words:
        clean = w.lower().strip(".,!?")

        if clean in suspicious_words:
            highlighted.append(f"<span style='color:red;font-weight:bold'>{w}</span>")
            found.append(w)
        else:
            highlighted.append(w)

    return " ".join(highlighted), found