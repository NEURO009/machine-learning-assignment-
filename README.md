# 🩺 Breast Cancer Classification using Machine Learning

## 📌 Project Overview

This project implements and compares five supervised Machine Learning algorithms for classifying breast tumors as **Benign (Non-Cancerous)** or **Malignant (Cancerous)** using the **Breast Cancer Wisconsin Diagnostic Dataset**.

The project demonstrates the complete Machine Learning workflow, including:

- Data preprocessing
- Model training
- Model evaluation
- Performance comparison
- Deployment using Streamlit Community Cloud

The deployed web application allows users to upload a test dataset, select a trained model, visualize performance metrics, and download prediction results.

---

# 🎯 Problem Statement

Breast cancer is one of the leading causes of cancer-related deaths among women worldwide. Early and accurate diagnosis plays a vital role in improving patient survival rates and reducing unnecessary treatments.

The objective of this project is to develop an end-to-end Machine Learning classification system capable of predicting whether a breast tumor is:

- **Benign (Non-Cancerous)**
- **Malignant (Cancerous)**

using diagnostic features extracted from breast tissue samples.

The project compares multiple supervised Machine Learning algorithms to determine the best-performing model based on several evaluation metrics.

---

# 📂 Dataset Description

### Dataset Name

Breast Cancer Wisconsin (Diagnostic) Dataset

### Source

https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data

### Dataset Information

- Total Samples: **569**
- Total Features: **30 Numerical Features**
- Target Variable: **diagnosis**
- Classification Type: **Binary Classification**

### Target Classes

| Value | Class |
|-------|--------|
| 0 | Benign |
| 1 | Malignant |

The dataset contains various numerical measurements extracted from digitized images of breast mass cell nuclei.

---

# 🧹 Data Preprocessing

The following preprocessing steps were performed:

- Removed unnecessary **ID** column
- Removed empty **Unnamed: 32** column
- Encoded target variable
  - Benign → 0
  - Malignant → 1
- Performed Train-Test Split (80:20)
- Used Stratified Sampling to preserve class distribution

---

# 🤖 Machine Learning Models Used

The following Machine Learning models were implemented and compared:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

---

# 📊 Model Comparison

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|-----------|---------:|----:|----------:|-------:|---------:|----:|
| Logistic Regression | **0.9649** | **0.9960** | **0.9750** | **0.9286** | **0.9512** | **0.9245** |
| Decision Tree | **0.9298** | **0.9246** | **0.9048** | **0.9048** | **0.9048** | **0.8492** |
| K-Nearest Neighbors (KNN) | **0.9561** | **0.9823** | **0.9744** | **0.9048** | **0.9383** | **0.9058** |
| Naive Bayes | **0.9386** | **0.9934** | **1.0000** | **0.8333** | **0.9091** | **0.8715** |
| Random Forest (Ensemble) | **0.9737** | **0.9929** | **1.0000** | **0.9286** | **0.9630** | **0.9442** |

---

# 📈 Model Performance Observations

| ML Model | Observation about Model Performance |
|-----------|-------------------------------------|
| **Logistic Regression** | Achieved excellent overall performance with high accuracy, precision, recall, and ROC-AUC. It serves as a strong baseline model for this dataset. |
| **Decision Tree** | Produced good classification results but achieved the lowest overall accuracy and MCC among all models. The model is more susceptible to overfitting. |
| **K-Nearest Neighbors (KNN)** | Demonstrated strong classification performance with high precision and accuracy. However, its recall was slightly lower than Logistic Regression and Random Forest. |
| **Naive Bayes** | Achieved perfect precision (1.0000), indicating no false positive predictions for malignant cases. However, its lower recall suggests that some malignant tumors were missed. |
| **Random Forest (Ensemble)** | Delivered the highest overall performance across nearly all evaluation metrics. It achieved the best balance between accuracy, precision, recall, F1-score, MCC, and ROC-AUC, making it the most reliable model for this dataset. |

---

# 🏆 Overall Winner

## Random Forest (Ensemble)

Random Forest achieved the highest overall performance on the Breast Cancer Wisconsin Dataset.

### Reasons

- Highest Accuracy (**97.37%**)
- Perfect Precision (**100%**)
- Highest F1 Score
- Highest Matthews Correlation Coefficient (MCC)
- Excellent ROC-AUC Score
- Best overall balance between false positives and false negatives

Therefore, **Random Forest (Ensemble)** is selected as the best-performing model for this classification task.

---

# 📊 Evaluation Metrics

The following evaluation metrics were used to compare the Machine Learning models:

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
├── README.md
├── requirements.txt
│
├── dataset/
│     ├── data.csv
│     └── test_data.csv
│
├── models/
│     ├── logistic.pkl
│     ├── decision_tree.pkl
│     ├── knn.pkl
│     ├── naive_bayes.pkl
│     └── random_forest.pkl
│
├── notebooks/
│     └── analysis.ipynb
│
└── outputs/
      ├── metrics.csv
      ├── logistic_regression_cm.png
      ├── decision_tree_cm.png
      ├── knn_cm.png
      ├── naive_bayes_cm.png
      └── random_forest_cm.png
```

---

# 🚀 How to Run the Project

## Clone the Repository

```bash
git clone https://github.com/NEURO009/machine-learning-assignment-.git
```

## Navigate to Project

```bash
cd machine-learning-assignment-
```

## Create Virtual Environment

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train Models

```bash
python train.py
```

## Evaluate Models

```bash
python evaluate.py
```

## Run the Streamlit Application

```bash
streamlit run app.py
```

---

# 🌐 GitHub Repository

**Repository Link**

https://github.com/NEURO009/machine-learning-assignment-

---

# 🚀 Streamlit Community Cloud Deployment

**Live Application**

https://esvuxhue4ozvce6yntqtza.streamlit.app

The Streamlit application provides the following functionality:

- Upload test dataset (.csv)
- Select one of the trained Machine Learning models
- Generate predictions
- View evaluation metrics
- Display confusion matrix
- Download prediction results

---

# 📸 Application Screenshots

## Home Page

*(Add Home Page Screenshot)*

## Model Performance

*(Add Metrics Screenshot)*

## Confusion Matrix

*(Add Confusion Matrix Screenshot)*

## Prediction Results

*(Add Prediction Results Screenshot)*

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

