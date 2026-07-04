import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# ─────────────────────────────────────────
# Load & Clean
# ─────────────────────────────────────────
df = pd.read_csv("appstore_games.csv")

df['Price']            = pd.to_numeric(df['Price'], errors='coerce').fillna(df['Price'].median())
df['Size']             = df['Size'].fillna(df['Size'].median())
df['In-app Purchases'] = pd.to_numeric(df['In-app Purchases'], errors='coerce').fillna(0)
df['Languages']        = df['Languages'].fillna('EN')
df = df.dropna(subset=['Average User Rating'])

df.drop(columns=['URL', 'Icon URL', 'Description', 'Name', 'Developer',
                 'Languages', 'Primary Genre', 'Subtitle',
                 'Original Release Date', 'Current Version Release Date', 'ID'],
        inplace=True)

df = pd.get_dummies(df, columns=['Genres', 'Age Rating'])
df = df.astype({col: int for col in df.select_dtypes(bool).columns})

# ─────────────────────────────────────────
# Correlation check — always do this first
# ─────────────────────────────────────────
print("Max correlation with target:")
print(df.corr()['Average User Rating'].abs().sort_values(ascending=False).head(10))

# ─────────────────────────────────────────
# Model
# ─────────────────────────────────────────
X = df.drop(columns=['Average User Rating'])
y = df['Average User Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

print(f"\nR² score: {model.score(X_test, y_test):.4f}")
# Result: 0.0417 — model learned nothing. Dataset has no real signal.
