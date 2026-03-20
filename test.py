from PIL import Image
import torch
from torchvision import transforms
from torch import nn

# Model structure (same as training)
model = nn.Sequential(
    nn.Flatten(),
    nn.Linear(128 * 128 * 3, 128),
    nn.ReLU(),
    nn.Linear(128, 2)
)

# Load trained weights
model.load_state_dict(torch.load("model.pth"))
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load test image
img = Image.open("test.jpg")
img = transform(img).unsqueeze(0)

# Predict
output = model(img)
_, predicted = torch.max(output, 1)

if predicted.item() == 1:
    print("Fake Image")
else:
    print("Real Image")