import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from model.text_model import predict_text
from model.image_model import predict_image
from model.fusion import fuse_predictions
import os

# Load dataset
fake = pd.read_csv("dataset/Fake.csv")
true = pd.read_csv("dataset/True.csv")

fake["label"] = "Fake"
true["label"] = "Real"

data = pd.concat([fake, true]).sample(frac=1).reset_index(drop=True)

texts = data["text"].values[:500]
true_labels = data["label"].values[:500]

image_folder_fake = "image_dataset/Fake"
image_folder_real = "image_dataset/Real"

predictions = []

print("Running multimodal evaluation...\n")

for i, text in enumerate(texts):

    # text prediction
    text_label, text_score = predict_text(text)

    # pick an image
    if true_labels[i] == "Fake":
        image_path = os.path.join(
            image_folder_fake,
            os.listdir(image_folder_fake)[i % len(os.listdir(image_folder_fake))]
        )
    else:
        image_path = os.path.join(
            image_folder_real,
            os.listdir(image_folder_real)[i % len(os.listdir(image_folder_real))]
        )

    # image prediction
    image_label, image_score = predict_image(image_path)

    # fusion prediction
    final_label, _ = fuse_predictions(
        text_label, text_score,
        image_label, image_score
    )

    predictions.append(final_label)

# Metrics
accuracy = accuracy_score(true_labels, predictions)
precision = precision_score(true_labels, predictions, pos_label="Fake")
recall = recall_score(true_labels, predictions, pos_label="Fake")
f1 = f1_score(true_labels, predictions, pos_label="Fake")

print("Accuracy :", accuracy)
print("Precision:", precision)
print("Recall   :", recall)
print("F1 Score :", f1)

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake","Real"],
            yticklabels=["Fake","Real"])

plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()