import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import datasets, transforms, models

# -----------------------------
# DATASET
# -----------------------------
class FaceDataset(Dataset):
    def __init__(self, root, transform=None):
        self.dataset = datasets.ImageFolder(root=root)
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img_np = np.array(img.resize((224, 224)))  # resize to backbone input
        if self.transform:
            img_np = self.transform(img_np)
        return img_np, label

# -----------------------------
# TRANSFORMS
# -----------------------------
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.RandomErasing(p=0.3),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# -----------------------------
# DEVICE
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -----------------------------
# LOAD DATASET
# -----------------------------
dataset_path = "dataset"  # replace with your dataset path
full_dataset = FaceDataset(dataset_path)
print("Classes:", full_dataset.dataset.class_to_idx)
print("Total images:", len(full_dataset))

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_indices, val_indices = random_split(range(len(full_dataset)), [train_size, val_size])

train_dataset = FaceDataset(dataset_path, transform=train_transform)
val_dataset = FaceDataset(dataset_path, transform=val_transform)

train_dataset = torch.utils.data.Subset(train_dataset, train_indices.indices)
val_dataset = torch.utils.data.Subset(val_dataset, val_indices.indices)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

# -----------------------------
# MODEL
# -----------------------------
# Using ResNet-18 backbone

model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
for param in model.parameters():
    param.requires_grad = True
model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(device)

# -----------------------------
# LOSS + OPTIMIZER
# -----------------------------
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)

# -----------------------------
# TRAINING
# -----------------------------
best_val_acc = 0.0
epochs = 25

for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(train_loader)

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            predicted = torch.argmax(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    val_acc = correct / total if total > 0 else 0
    scheduler.step()

    print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:.4f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "efficientnet_deepfake_best.pth")
        print(f"  ✅ Best model saved (acc={val_acc:.4f})")

print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.4f}")