import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

# ─────────────────────────────────────────
# Transforms
# ─────────────────────────────────────────
# X-rays are grayscale so we convert to single channel.
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# ─────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────
train_set = ImageFolder(root="data/train", transform=transform)
test_set  = ImageFolder(root="data/test",  transform=transform)

print(f"Train: {len(train_set)} | Test: {len(test_set)}")
print(f"Classes: {len(train_set.classes)}")
# ~989 images across 10 classes — far too small to train from scratch

train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_set,  batch_size=64, shuffle=False)

# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
class BoneCNN(nn.Module):
    def __init__(self):
        super(BoneCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=4)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1   = nn.Linear(128 * 53 * 53, 128)
        self.fc2   = nn.Linear(128, 64)
        self.fc3   = nn.Linear(64, 32)
        self.fc4   = nn.Linear(32, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model     = BoneCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# ─────────────────────────────────────────
# Training
# ─────────────────────────────────────────
for epoch in range(30):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch {epoch+1:2d} | Loss: {running_loss/len(train_loader):.4f}")

# ─────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────
model.eval()
correct = total = 0

with torch.no_grad():
    for x_batch, y_batch in test_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        _, predicted = torch.max(model(x_batch), 1)
        total   += y_batch.size(0)
        correct += (predicted == y_batch).sum().item()

print(f"\nAccuracy: {100 * correct / total:.2f}%")
# Result: 26.43% — better than random (10%) but far from useful.
# Root causes:
# 1. ~989 images for 10 classes is not enough to train from scratch
# 2. Medical X-rays with doctor annotations (arrows, numbers) add noise
# 3. Fine-grained fracture types need stronger pretrained features
# Solution: Transfer Learning with ResNet or similar pretrained model
