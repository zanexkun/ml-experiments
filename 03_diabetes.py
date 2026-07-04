import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# ─────────────────────────────────────────
# Load & Clean
# ─────────────────────────────────────────
df = pd.read_csv("Diabetes Classification.csv")

df['Gender']                    = df['Gender'].map({'Female': 0, 'Male': 1})
df['Family History of Diabetes'] = df['Family History of Diabetes'].map({'No': 0, 'Yes': 1})
df['Smoking']                   = df['Smoking'].map({'No': 0, 'Yes': 1})
df['Diet']                      = df['Diet'].map({'Healthy': 1, 'Poor': 0})
df['Exercise']                  = df['Exercise'].map({'Regular': 1, 'No': 0})
df['Diagnosis']                 = df['Diagnosis'].map({'Yes': 1, 'No': 0})

oe = OrdinalEncoder(categories=[['Low', 'Normal', 'High']])
df['Blood Pressure'] = oe.fit_transform(df[['Blood Pressure']])

print(f"Dataset size: {len(df)} rows")
print(f"Class balance:\n{df['Diagnosis'].value_counts()}")
# Only 128 rows — far too small for a reliable medical classifier

# ─────────────────────────────────────────
# PyTorch Model
# ─────────────────────────────────────────
X = df.drop('Diagnosis', axis=1)
y = df['Diagnosis']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

X_train_tensor = torch.from_numpy(X_train).float()
X_test_tensor  = torch.from_numpy(X_test).float()
y_train_tensor = torch.from_numpy(y_train.values).float().unsqueeze(1)
y_test_tensor  = torch.from_numpy(y_test.values).float().unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_tensor, y_train_tensor), batch_size=32, shuffle=True)


class DiabetesNet(nn.Module):
    def __init__(self):
        super(DiabetesNet, self).__init__()
        self.fc1 = nn.Linear(10, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return torch.sigmoid(self.fc3(x))


model     = DiabetesNet()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(100):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        loss = criterion(model(x_batch), y_batch)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

model.eval()
with torch.no_grad():
    preds       = model(X_test_tensor)
    predictions = (preds >= 0.5).float()
    accuracy    = (predictions == y_test_tensor).float().mean().item()
    print(f"\nPyTorch accuracy: {accuracy:.4f}")
    # Result: 100% — suspicious on a 128-row medical dataset

# ─────────────────────────────────────────
# Cross-validation confirms the problem
# ─────────────────────────────────────────
rf     = RandomForestClassifier(random_state=42)
scores = cross_val_score(rf, X, y, cv=5)
print(f"\nRandom Forest cross-val scores: {scores}")
print(f"Mean: {scores.mean():.4f}")
# 97.6% mean but one fold drops to 88% — dataset too small and synthetic.
# 100% accuracy on real medical data would be extraordinary.
# This dataset has artificially clean separations between classes.
