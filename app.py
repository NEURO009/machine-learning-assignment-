import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

# PAGE CONFIGURATION

st.set_page_config(
    page_title="Breast Cancer Classification System",
    page_icon="🩺",
    layout="wide"
)

# TITLE

st.title("🩺 Breast Cancer Classification System")

st.markdown("""
Welcome to the **Breast Cancer Classification System**, a Machine Learning application
designed to classify breast tumors as **Benign (Non-Cancerous)** or
**Malignant (Cancerous)** using multiple supervised learning algorithms.

This application demonstrates the complete machine learning workflow including
prediction, evaluation, and performance comparison of different classification models.
""")

# ABOUT PROJECT

with st.expander("📖 About this Project"):

    st.markdown("""
### Project Objective

The objective of this project is to compare different Machine Learning algorithms
for predicting breast cancer diagnosis using the **Breast Cancer Wisconsin Dataset**.

The application allows users to upload a test dataset,
choose a trained model, evaluate its performance,
and visualize the prediction results.

---

### Machine Learning Models Implemented

- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Gaussian Naive Bayes
- Random Forest

---

### Performance Metrics

The application displays:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

---

### Workflow

1. Select a trained model.
2. Upload **test_data.csv**.
3. Generate predictions.
4. View evaluation metrics.
5. Visualize confusion matrix.
6. Download prediction results.
""")

# SIDEBAR

st.sidebar.title("⚙️ Control Panel")

st.sidebar.markdown("""
### Steps

1️⃣ Select a Machine Learning Model

2️⃣ Upload **test_data.csv**

3️⃣ View Model Performance

4️⃣ Download Prediction Results
""")

# LOAD MODELS

models = {
    "Logistic Regression": (joblib.load("models/logistic.pkl")),
    "Decision Tree": (joblib.load("models/decision_tree.pkl")),
    "K-Nearest Neighbors (KNN)": (joblib.load("models/knn.pkl")),
    "Naive Bayes": (joblib.load("models/naive_bayes.pkl")),
    "Random Forest": (joblib.load("models/random_forest.pkl"))
}


# SIDEBAR INPUTS

selected_model = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload test_data.csv",
    type=["csv"]
)

# MAIN APPLICATION

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.header("📄 Uploaded Dataset")

    col1, col2 = st.columns(2)

    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])

    st.dataframe(df.head(10), use_container_width=True)

    # Separate Features & Target

    X = df.drop("diagnosis", axis=1)
    y = df["diagnosis"]

    model = models[selected_model]
    X_input = X

    predictions = model.predict(X_input)
    probabilities = model.predict_proba(X_input)[:, 1]

    # METRICS

    accuracy = accuracy_score(y, predictions)
    precision = precision_score(y, predictions)
    recall = recall_score(y, predictions)
    f1 = f1_score(y, predictions)
    auc = roc_auc_score(y, probabilities)
    mcc = matthews_corrcoef(y, predictions)

    st.header("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("Precision", f"{precision:.4f}")
    c3.metric("Recall", f"{recall:.4f}")

    c4, c5, c6 = st.columns(3)

    c4.metric("F1 Score", f"{f1:.4f}")
    c5.metric("ROC-AUC", f"{auc:.4f}")
    c6.metric("MCC", f"{mcc:.4f}")

    # CONFUSION MATRIX

    st.header("📈 Confusion Matrix")

    cm = confusion_matrix(y, predictions)

    fig, ax = plt.subplots(figsize=(6,5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Benign", "Malignant"],
        yticklabels=["Benign", "Malignant"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title(selected_model)

    st.pyplot(fig)

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    st.header("🔍 Prediction Results")

    results = df.copy()

    results["Predicted Diagnosis"] = predictions

    st.dataframe(results, use_container_width=True)

    # --------------------------------------------------------
    # DOWNLOAD BUTTON
    # --------------------------------------------------------

    csv = results.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="📥 Download Prediction Results",
        data=csv,
        file_name="prediction_results.csv",
        mime="text/csv"
    )

else:

    st.info("👈 Select a model from the sidebar and upload **test_data.csv** to begin evaluation.")

# ------------------------------------------------------------
# FOOTER
# ------------------------------------------------------------

st.markdown("---")

st.markdown(
    """
**Machine Learning Assignment**

Breast Cancer Classification using Logistic Regression, Decision Tree,
K-Nearest Neighbors, Naive Bayes, and Random Forest.

Developed using **Python**, **Scikit-learn**, and **Streamlit**.
"""
)