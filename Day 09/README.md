# Day 9 - Iris Flower Classification System

## Overview

This project is part of my Machine Learning Internship (Day 9). The goal is to build a classification model that predicts the species of an Iris flower based on its physical features using the Logistic Regression algorithm. The project also demonstrates how to evaluate a classification model using different evaluation metrics.

---

# What is Classification?

Classification is a type of Machine Learning task in which the model predicts a category or class instead of a numerical value.

Examples:

* Spam or Not Spam Email
* Disease Detection
* Face Recognition
* Iris Flower Species Prediction

In this project, the model predicts one of the following Iris flower species:

* Setosa
* Versicolor
* Virginica

---

# Difference Between Regression and Classification

| Regression                      | Classification                                 |
| ------------------------------- | ---------------------------------------------- |
| Predicts numerical values       | Predicts categories or classes                 |
| Output is continuous            | Output is discrete                             |
| Example: House Price Prediction | Example: Flower Species Prediction             |
| Algorithms: Linear Regression   | Algorithms: Logistic Regression, Decision Tree |

---

# Dataset

This project uses the **Iris Dataset** provided by Scikit-learn.

Dataset Information:

* Total Samples: 150
* Features: 4

  * Sepal Length
  * Sepal Width
  * Petal Length
  * Petal Width
* Target Classes:

  * Setosa
  * Versicolor
  * Virginica

---

# Technologies Used

* Python
* Scikit-learn
* Pandas
* Matplotlib
* Seaborn

---

# Project Workflow

1. Load the Iris dataset.
2. Explore the dataset.
3. Split the data into training and testing sets.
4. Train a Logistic Regression model.
5. Make predictions.
6. Evaluate the model.
7. Generate a Confusion Matrix.
8. Display sample predictions.

---

# Evaluation Metrics Used

### Accuracy

Measures the overall percentage of correct predictions.

### Precision

Measures how many predicted positive classifications are actually correct.

### Recall

Measures how many actual positive samples are correctly identified.

### F1-Score

The harmonic mean of Precision and Recall. It provides a balanced measure of model performance.

### Confusion Matrix

A table that compares actual values with predicted values and helps identify classification errors.

---

# Model Performance

The Logistic Regression model achieved excellent performance on the Iris dataset.

Observed Metrics:

* High Accuracy
* High Precision
* High Recall
* High F1-Score

The Confusion Matrix showed that most flower species were classified correctly with very few or no misclassifications.

---

# Files Included

```text
Day-9/
│
├── classification_practice.py
├── iris_project.py
├── README.md
└── screenshots/
    └── confusion_matrix.png
```

---

# Learning Outcomes

After completing this project, I learned:

* What Classification is.
* Difference between Regression and Classification.
* How to train a Logistic Regression model.
* How to split a dataset into training and testing sets.
* How to evaluate a classification model.
* How to interpret Accuracy, Precision, Recall, F1-Score, and the Confusion Matrix.
* How to make predictions using a trained Machine Learning model.

---

# Future Improvements

* Compare Logistic Regression with Decision Tree.
* Test additional classification algorithms.
* Deploy the model using Streamlit.
* Experiment with larger classification datasets.

---
