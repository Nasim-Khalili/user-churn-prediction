# User Churn Prediction (Machine Learning Project)

## 📌 Project Overview

This project focuses on building an **end-to-end Machine Learning pipeline** to predict **customer churn** using structured/tabular data. The goal is to identify customers who are likely to leave a service, which is a critical business problem in telecom, SaaS, and subscription-based companies.

The project is designed as a **portfolio-ready project** for a **Machine Learning Engineer** role and demonstrates skills in:

* Data preprocessing
* Feature engineering
* Model training & evaluation
* Model persistence
* Clean project structure
* Git & production-oriented practices

---

## 🧠 Problem Statement

Customer churn refers to customers who stop using a company's service. Predicting churn helps businesses:

* Reduce revenue loss
* Improve customer retention strategies
* Target high-risk customers proactively

This project formulates churn prediction as a **binary classification problem**.

---

## 📂 Project Structure

```
user-churn-prediction/
│
├── data/
│   └── raw_churn.csv              # Raw dataset (not tracked in Git)
│
├── notebooks/
│   └── EDA.ipynb                  # Exploratory Data Analysis
│
├── src/
│   ├── preprocess.py              # Data cleaning & feature engineering
│   ├── train.py                   # Model training script
│   └── evaluate.py                # Model evaluation
│
├── api/
│   └── app.py                     # (Optional) FastAPI inference API
│
├── models/
│   └── churn_model.pkl            # Trained model (optional)
│
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **Programming Language:** Python
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-learn, XGBoost
* **Model Persistence:** Joblib
* **API (Optional):** FastAPI
* **Version Control:** Git & GitHub

---

## 🔍 Dataset

* Source: Telco Customer Churn Dataset
* Type: Tabular data
* Target variable: `Churn` (0 = No, 1 = Yes)

⚠️ Note: Large datasets are excluded from GitHub and handled locally.

---

## 🧪 Machine Learning Pipeline

1. **Data Loading**
2. **Data Cleaning**

   * Handling missing values
   * Type conversions
3. **Feature Engineering**

   * Encoding categorical variables
   * Scaling numerical features
4. **Train/Test Split**
5. **Model Training**

   * XGBoost Classifier
6. **Model Evaluation**

   * Accuracy
   * Precision / Recall
   * Confusion Matrix
7. **Model Saving**

---

## 📊 Model Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics help assess performance on imbalanced churn data.

---

## 🚀 How to Run the Project

### 1️⃣ Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run Preprocessing

```bash
python src/preprocess.py
```

### 4️⃣ Train Model

```bash
python src/train.py
```

### 5️⃣ Evaluate Model

```bash
python src/evaluate.py
```

---

## 📈 Future Improvements

* Hyperparameter tuning (GridSearch / Optuna)
* Cross-validation
* Handling class imbalance (SMOTE)
* Model explainability (SHAP)
* Full deployment with FastAPI + Docker

---

## 👩‍💻 Author

**Nasim Khalili**
Machine Learning / Backend Enthusiast

---

## ⭐ Why This Project Matters

This project demonstrates:

* Real-world ML workflow
* Clean and scalable code structure
* Practical understanding of ML engineering concepts

It is suitable as a **strong portfolio project for Machine Learning Engineer roles**.

---

If you find this project useful, feel free to ⭐ the repository!
