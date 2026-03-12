from transformers import pipeline

# Load our trained BERT fake news model
classifier = pipeline(
    "text-classification",
    model="model/bert_fake_news",
    tokenizer="model/bert_fake_news"
)

def predict_text(text):

    result = classifier(text[:256])[0]

    label = result["label"]
    score = result["score"]

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