import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor()
])

# Load dataset
dataset = datasets.ImageFolder(
    root="image_dataset",
    transform=transform
)

train_loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True
)

print("Classes:", dataset.classes)

# Load pretrained ResNet50
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

# Modify final layer
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, 2)

# Loss function
criterion = nn.CrossEntropyLoss()

# Optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

# Training
epochs = 3

for epoch in range(epochs):

    running_loss = 0

    for images, labels in train_loader:

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch+1}/{epochs} Loss:", running_loss)

# Save model
torch.save(model.state_dict(), "model/image_cnn.pth")

print("Image model trained successfully!")