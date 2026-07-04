import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# ─────────────────────────────────────────
# Load & Clean
# ─────────────────────────────────────────
df = pd.read_csv("AI_Impact_on_Jobs_2030.csv")
df = df.drop(columns=['Employee_ID', 'Country', 'Average_Salary_USD', 'Required_Skills'])

oe = OrdinalEncoder(categories=[['PhD', 'Bachelor', 'High School', 'Master']])
df['Education_Level'] = oe.fit_transform(df[['Education_Level']])

df = pd.get_dummies(df, columns=[
    'Remote_Work_Possibility', 'Automation_Level', 'Company_Size',
    'AI_Tool_Usage', 'Upskilling_Needed', 'Hiring_Trend_2026',
    'Industry', 'Job_Title'
])
df = df.astype({col: int for col in df.select_dtypes(bool).columns})

# ─────────────────────────────────────────
# Correlation check — caught the problem here
# ─────────────────────────────────────────
print("Max correlation with target:")
print(df.corr()['AI_Replacement_Risk'].abs().sort_values(ascending=False).head(10))
# Max correlation: 0.04 — clear sign of fake data

# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
X = df.drop(columns=['AI_Replacement_Risk'])
y = df['AI_Replacement_Risk']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = torch.from_numpy(scaler.fit_transform(X_train)).float()
X_test_scaled  = torch.from_numpy(scaler.transform(X_test)).float()
y_train_tensor = torch.from_numpy(y_train.values).float().unsqueeze(1)
y_test_tensor  = torch.from_numpy(y_test.values).float().unsqueeze(1)

train_loader = DataLoader(TensorDataset(X_train_scaled, y_train_tensor), batch_size=64, shuffle=True)


class JobsNet(nn.Module):
    def __init__(self, input_size):
        super(JobsNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 180)
        self.fc2 = nn.Linear(180, 60)
        self.fc3 = nn.Linear(60, 10)
        self.fc4 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


model     = JobsNet(X_train.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(30):
    model.train()
    running_loss = 0.0
    for x_batch, y_batch in train_loader:
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
with torch.no_grad():
    pred   = model(X_test_scaled).numpy().flatten()
    y_true = y_test_tensor.numpy().flatten()

print(f"\nR²: {r2_score(y_true, pred):.4f}")
print(f"MAE: {abs(pred - y_true).mean():.4f}")
# Result: R² = -0.81 — worse than predicting the mean.
# Confirmed fake dataset despite Kaggle gold badge.
