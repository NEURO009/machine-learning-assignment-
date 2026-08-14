"""Train the ML models for the assignment."""
import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

# ----------------------------------------------------
# Create folders if they don't exist
# ----------------------------------------------------

os.makedirs("models", exist_ok=True)

# ----------------------------------------------------
# Load Dataset
# ----------------------------------------------------

df = pd.read_csv("dataset/data.csv")

# ----------------------------------------------------
# Data Preprocessing
# ----------------------------------------------------

df.drop(columns=["id", "Unnamed: 32"], inplace=True, errors="ignore")

df["diagnosis"] = df["diagnosis"].map({
    "B": 0,
    "M": 1
})

X = df.drop("diagnosis", axis=1)
y = df["diagnosis"]

# ----------------------------------------------------
# Train-Test Split
# ----------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ----------------------------------------------------
# Save Test Dataset
# ----------------------------------------------------

test_data = X_test.copy()
test_data["diagnosis"] = y_test

test_data.to_csv("test_data.csv", index=False)

# ----------------------------------------------------
# Feature Scaling
# ----------------------------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, "models/scaler.pkl")

# ----------------------------------------------------
# Initialize Models
# ----------------------------------------------------

logistic_model = LogisticRegression(random_state=42)

decision_tree_model = DecisionTreeClassifier(random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)

naive_bayes_model = GaussianNB()

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# ----------------------------------------------------
# Train Models
# ----------------------------------------------------

logistic_model.fit(X_train_scaled, y_train)

decision_tree_model.fit(X_train, y_train)

knn_model.fit(X_train_scaled, y_train)

naive_bayes_model.fit(X_train, y_train)

random_forest_model.fit(X_train, y_train)

# ----------------------------------------------------
# Save Models
# ----------------------------------------------------

joblib.dump(logistic_model, "models/logistic.pkl")

joblib.dump(decision_tree_model, "models/decision_tree.pkl")

joblib.dump(knn_model, "models/knn.pkl")

joblib.dump(naive_bayes_model, "models/naive_bayes.pkl")

joblib.dump(random_forest_model, "models/random_forest.pkl")

print("Training completed successfully.")
print("Models saved in models/")
print("test_data.csv created.")
print("Training pipeline is ready to be implemented.")
