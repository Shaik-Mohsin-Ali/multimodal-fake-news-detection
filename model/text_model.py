from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

classifier = None

tokenizer = None
model = None

def load_model():
    global tokenizer, model

    if model is None:
        tokenizer = AutoTokenizer.from_pretrained("model/bert_fake_news")
        model = AutoModelForSequenceClassification.from_pretrained("model/bert_fake_news")
        model.eval()

def predict_text(text):

    load_model()

    inputs = tokenizer(text[:256], return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)
    score, pred = torch.max(probs, 1)

    label = "LABEL_0" if pred.item() == 0 else "LABEL_1"
    score = score.item()

    # Convert HuggingFace labels
    if label == "LABEL_0":
        final_label = "Fake"
    else:
        final_label = "Real"

    return final_label, score


def explain_text(text):

    suspicious_words = [
        "breaking", "shocking", "secret", "unbelievable",
        "aliens", "ufo", "miracle", "hoax", "conspiracy"
    ]

    words = text.split()
    highlighted_text = []
    found_words = []

    for word in words:
        clean_word = word.lower().strip(".,!?")

        if clean_word in suspicious_words:
            highlighted_text.append(f"<span style='color:red;font-weight:bold'>{word}</span>")
            found_words.append(word)
        else:
            highlighted_text.append(word)

    highlighted_sentence = " ".join(highlighted_text)

    return highlighted_sentence, found_words