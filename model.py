import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Improved dataset
data = {
    "amount": [50, 120, 5000, 20000, 75000, 150, 300, 90000, 100000, 250],
    "location_change": [0, 0, 0, 1, 1, 0, 0, 1, 1, 0],
    "frequency": [1, 2, 3, 10, 20, 1, 2, 25, 30, 2],
    "is_fraud": [0, 0, 0, 1, 1, 0, 0, 1, 1, 0]
}

df = pd.DataFrame(data)

X = df.drop("is_fraud", axis=1)
y = df["is_fraud"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

# Save model
with open("fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("✅ Model trained and saved!")