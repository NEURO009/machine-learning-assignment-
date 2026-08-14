# ML Assignment 2

This repository contains the project structure for a machine learning task.

## Structure
- `app.py` — application entry point
- `train.py` — model training script
- `evaluate.py` — evaluation script
- `dataset/` — training datasets
- `models/` — saved model artifacts
- `outputs/` — metrics and plots
- `notebooks/` — exploratory notebooks

## Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Train the model: `python train.py`
3. Evaluate: `python evaluate.py`
# 🩺 Breast Cancer Classification using Machine Learning

## 📌 Project Overview

This project implements and compares five supervised Machine Learning algorithms for classifying breast tumors as **Benign** or **Malignant** using the **Breast Cancer Wisconsin Diagnostic Dataset**.

The project includes data preprocessing, model training, evaluation, and deployment through a **Streamlit web application**, allowing users to upload test data and compare the performance of multiple trained models.

---

# 🎯 Problem Statement

Breast cancer is one of the most common cancers worldwide. Early detection significantly improves treatment outcomes.

The objective of this project is to develop a machine learning-based classification system capable of predicting whether a breast tumor is:

- Benign (Non-Cancerous)
- Malignant (Cancerous)

The application compares multiple machine learning algorithms based on different evaluation metrics.

---

# 📂 Dataset

**Dataset Name**

Breast Cancer Wisconsin (Diagnostic) Dataset

**Source**

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

**Target Variable**

```
diagnosis
```

Values:

- B → Benign
- M → Malignant

Dataset contains:

- 569 samples
- 30 numerical features
- Binary classification problem

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Removed ID column
- Removed empty "Unnamed: 32" column
- Encoded target variable
  - Benign → 0
  - Malignant → 1
- Train-Test Split (80:20)
- Stratified sampling
- Feature Scaling using StandardScaler (for Logistic Regression and KNN)

---

# 🤖 Machine Learning Models

The following models were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest

---

# 📊 Evaluation Metrics

Each model is evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC Score
- Matthews Correlation Coefficient (MCC)
- Confusion Matrix

---

# 📁 Project Structure

```
machine-learning-assignment/
│
├── app.py
├── train.py
├── evaluate.py
├── analysis.ipynb
├── README.md
├── requirements.txt
├── test_data.csv
│
├── dataset/
│     └── data.csv
│
├── models/
│     ├── logistic.pkl
│     ├── decision_tree.pkl
│     ├── knn.pkl
│     ├── naive_bayes.pkl
│     ├── random_forest.pkl
│     └── scaler.pkl
│
├── outputs/
│     ├── metrics.csv
│     ├── logistic_regression_cm.png
│     ├── decision_tree_cm.png
│     ├── knn_cm.png
│     ├── naive_bayes_cm.png
│     └── random_forest_cm.png
```

---

# 🚀 How to Run

## 1. Clone Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_LINK>
```

---

## 2. Navigate to Project

```bash
cd machine-learning-assignment
```

---

## 3. Create Virtual Environment

### Windows

```bash
python -m venv .venv
```

Activate

```bash
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
```

Activate

```bash
source .venv/bin/activate
```

---

## 4. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 5. Train Models

```bash
python train.py
```

---

## 6. Evaluate Models

```bash
python evaluate.py
```

---

## 7. Launch Streamlit Application

```bash
streamlit run app.py
```

---

# 🌐 Streamlit Application

The application allows users to:

- Upload a CSV test dataset
- Select any trained model
- Generate predictions
- View evaluation metrics
- Visualize the confusion matrix
- Download prediction results

---

# 📈 Results

The trained models are compared using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- MCC

The comparison results are automatically saved in:

```
outputs/metrics.csv
```

---

# 📸 Application Screenshots

## Home Page

(Add Screenshot)

---

## Model Evaluation

(Add Screenshot)

---

## Confusion Matrix

(Add Screenshot)

---

## Prediction Results

(Add Screenshot)

---

# 🛠 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

---

# 👨‍💻 Author

**Name:** Navneet Sharma

**Course:** M.Tech

**Subject:** Machine Learning Assignment

---

# 📄 License

This project is developed for educational purposes as part of an academic assignment.