import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import cv2
import numpy as np

# Load pretrained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# Modify final layer for 2 classes
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Load trained weights if available
try:
    state_dict = torch.load("model/image_cnn.pth", map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.to(torch.device("cpu"))
except Exception as e:
    print("Image model load error:", e)

model.eval()

model.to(torch.device("cpu"))
model.eval()
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

classes = ["Fake", "Real"]

def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")
    img = transform(image).unsqueeze(0)

    with torch.no_grad():
        outputs = model(img)
        probs = torch.softmax(outputs, dim=1)
        score, pred = torch.max(probs, 1)

    label = classes[pred.item()]
    return label, score.item()

def generate_gradcam(image_path):

    image = Image.open(image_path).convert("RGB")
    img_tensor = transform(image).unsqueeze(0)
    img_tensor.requires_grad = True

    # Get final convolution layer
    target_layer = model.layer4[-1]

    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0])

    handle_forward = target_layer.register_forward_hook(forward_hook)
    handle_backward = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    pred_class = output.argmax()

    model.zero_grad()
    output[0, pred_class].backward()

    grads = gradients[0]
    acts = activations[0]

    weights = grads.mean(dim=(2, 3), keepdim=True)
    cam = (weights * acts).sum(dim=1).squeeze()

    cam = torch.relu(cam)
    cam = cam / cam.max()

    cam = cam.detach().numpy()

    heatmap = cv2.resize(cam, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    original = cv2.imread(image_path)
    original = cv2.resize(original, (224, 224))

    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)

    heatmap_path = "static/gradcam.jpg"
    cv2.imwrite(heatmap_path, superimposed)

    handle_forward.remove()
    handle_backward.remove()

    return heatmap_path