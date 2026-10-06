# Diabetes Prediction Using Machine Learning

## Project Overview

This project focuses on predicting diabetes using machine learning algorithms based on the Pima Indians Diabetes Dataset.

The project is inspired by the research paper:

**Chang, V., Bailey, J., Xu, Q. A., & Sun, Z. (2022).**  
*Pima Indians diabetes mellitus classification based on machine learning (ML) algorithms.*  
Neural Computing and Applications.

The project implements multiple machine learning algorithms and compares their performance. A Streamlit-based web application is also developed to allow users to enter patient information and obtain a model prediction.

---

## Research Paper-Based Implementation

### Dataset

The project uses the **Pima Indians Diabetes Dataset**, which contains:

- 768 patient records
- 8 input features
- 1 target variable
- Binary classification: diabetic or non-diabetic

### Input Features

1. Pregnancies
2. Glucose
3. Blood Pressure
4. Skin Thickness
5. Insulin
6. BMI
7. Diabetes Pedigree Function
8. Age

### Target Variable

**Outcome**

- `0` → Non-diabetic
- `1` → Diabetic

---

## Data Preprocessing

The dataset contains zero values in some medical measurements where zero is not physiologically meaningful.

The following features were processed:

- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI

Zero values were replaced with missing values and then replaced using median imputation.

The dataset was divided into training and testing sets using an 80:20 split.

---

## Machine Learning Algorithms

Three machine learning algorithms were implemented:

### 1. Random Forest

Random Forest is an ensemble learning algorithm that combines multiple decision trees to improve prediction performance.

### 2. Naïve Bayes

Naïve Bayes is a probabilistic classification algorithm based on Bayes' theorem.

### 3. Decision Tree

Decision Tree is a supervised learning algorithm that makes predictions using a sequence of decision rules.

---

## Research Paper Comparison

| Aspect | Research Paper | Our Implementation |
|---|---|---|
| Dataset | Pima Indians Diabetes Dataset | Same |
| Records | 768 | 768 |
| Features | 8 | 8 |
| Target | Diabetes Outcome | Same |
| Models | Naïve Bayes, Random Forest, J48 | Naïve Bayes, Random Forest, Decision Tree |
| Programming Language | R | Python |
| Invalid Zero Handling | Median Imputation | Median Imputation |
| Evaluation | Accuracy, Precision, Sensitivity, Specificity, F-score, AUC | Accuracy, Precision, Recall, F1-score |
| Application | E-diagnosis / IoMT concept | Streamlit Prediction Application |

---

## Research Paper Results

The research paper reported the following results on the full dataset after preprocessing:

| Model | Accuracy |
|---|---:|
| J48 Decision Tree | 74.78% |
| Random Forest | 79.57% |
| Naïve Bayes | 78.67% |

Random Forest achieved the highest accuracy among the three models on the full dataset in the research paper.

---
## Our Model Results

The three machine learning models were evaluated using Accuracy, Precision, Recall, and F1-Score.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Random Forest | 77.92% | 72.73% | 59.26% | 65.31% |
| Naïve Bayes | 70.13% | 56.67% | 62.96% | 59.65% |
| Decision Tree | 68.18% | 55.32% | 48.15% | 51.49% |

### Best Performing Model

Random Forest achieved the highest accuracy among the three implemented models, with an accuracy of **77.92%**.

The research paper reported an accuracy of **79.57%** for Random Forest on the full Pima Indians Diabetes Dataset. Our implementation achieved **77.92%**, showing a comparable performance under our experimental setup.

## Streamlit Application

A Streamlit web application was developed for interactive diabetes prediction.

The application allows the user to enter:

- Pregnancies
- Glucose
- Blood Pressure
- Skin Thickness
- Insulin
- BMI
- Diabetes Pedigree Function
- Age

The trained machine learning model then generates a prediction.

**Note:** This application is developed for educational and research purposes and should not be considered a medical diagnosis tool.

---

## Project Structure

```text
Diabetes-Prediction/
│
├── app/
│   └── app.py
│
├── data/
│   └── diabetes.csv
│
├── models/
│   ├── best_diabetes_model.pkl
│   ├── scaler.pkl
│   └── median_values.pkl
│
├── notebooks/
│   └── diabetes_prediction.ipynb
│
├── results/
│   └── model evaluation results and graphs
│
└── README.md