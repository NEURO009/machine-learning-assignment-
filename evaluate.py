"""Evaluate the trained ML models for the assignment."""
import os
import joblib
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------------------------
# Create output folder
# -----------------------------------------------------

os.makedirs("outputs", exist_ok=True)

# -----------------------------------------------------
# Load Test Data
# -----------------------------------------------------

test_df = pd.read_csv("test_data.csv")

X_test = test_df.drop("diagnosis", axis=1)
y_test = test_df["diagnosis"]

# -----------------------------------------------------
# Load Scaler
# -----------------------------------------------------

scaler = joblib.load("models/scaler.pkl")

X_test_scaled = scaler.transform(X_test)

# -----------------------------------------------------
# Load Models
# -----------------------------------------------------

models = {
    "Logistic Regression": (
        joblib.load("models/logistic.pkl"),
        True
    ),
    "Decision Tree": (
        joblib.load("models/decision_tree.pkl"),
        False
    ),
    "KNN": (
        joblib.load("models/knn.pkl"),
        True
    ),
    "Naive Bayes": (
        joblib.load("models/naive_bayes.pkl"),
        False
    ),
    "Random Forest": (
        joblib.load("models/random_forest.pkl"),
        False
    )
}

# -----------------------------------------------------
# Evaluate Function
# -----------------------------------------------------

results = []

for model_name, (model, use_scaled_data) in models.items():

    if use_scaled_data:
        X = X_test_scaled
    else:
        X = X_test

    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    results.append({
        "Model": model_name,
        "Accuracy": accuracy,
        "AUC": auc,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "MCC": mcc
    })

    # -------------------------------
    # Confusion Matrix
    # -------------------------------

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign","Malignant"],
        yticklabels=["Benign","Malignant"]
    )

    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()

    filename = model_name.lower().replace(" ", "_")

    plt.savefig(f"outputs/{filename}_cm.png")

    plt.close()

    # -------------------------------
    # Classification Report
    # -------------------------------

    print("="*60)
    print(model_name)
    print("="*60)

    print(classification_report(y_test, y_pred))

# -----------------------------------------------------
# Save Comparison Table
# -----------------------------------------------------

metrics_df = pd.DataFrame(results)

metrics_df.to_csv("outputs/metrics.csv", index=False)

print("\nEvaluation Complete!\n")
print("Evaluation pipeline is ready to be implemented.")
