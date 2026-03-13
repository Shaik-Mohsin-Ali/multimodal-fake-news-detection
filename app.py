from flask import Flask, render_template, request
import os

from model.text_model import predict_text, explain_text
from model.image_model import predict_image, generate_gradcam
from model.fusion import fuse_predictions

app = Flask(__name__)

UPLOAD_FOLDER = "static"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():

    result = None

    if request.method == "POST":
        text = request.form["text"]
        image = request.files.get("image")

        if not image:
            return render_template("index.html", error="Please upload an image")

        image_path = os.path.join(UPLOAD_FOLDER, image.filename)
        image.save(image_path)

        text_label, text_score = predict_text(text)
        image_label, image_score = predict_image(image_path)
        heatmap_path = generate_gradcam(image_path)
        highlighted_text, keywords = explain_text(text)

        final_label, final_score = fuse_predictions(
            text_label, text_score,
            image_label, image_score
        )

        result = {
            "text_label": text_label,
            "image_label": image_label,
            "final_label": final_label,
            "score": round(final_score * 100, 2),
            "keywords": keywords,
            "highlighted_text": highlighted_text,
            "heatmap": heatmap_path
        }

    return render_template("index.html", result=result)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)