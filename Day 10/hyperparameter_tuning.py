
# ================= Import Of required libraries ================

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
import pickle

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns


# ================ Load dataset ===============
cancer = load_breast_cancer()

X = cancer.data
y = cancer.target

# ================ Split data ====================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ================ Create model ==================
hyp_model = LogisticRegression(max_iter=50000)

baseline_model = LogisticRegression(max_iter=10000)
baseline_model.fit(X_train, y_train)

baseline_pred = baseline_model.predict(X_test)

# =========== Hyperparameters to test ============
param_grid = {
    "C": [0.01, 0.1, 1, 10, 100],
    "solver": ["liblinear", "lbfgs"]
}

# ================ Grid Search ================
grid_search = GridSearchCV(
    estimator=hyp_model,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

# =============== Train ======================
grid_search.fit(X_train, y_train)


# =============== Best trained model ==========
best_model = grid_search.best_estimator_

# Make predictions
y_pred = best_model.predict(X_test)


# ============= Evaluate ===============



print("\nTuned Model Performance")

print("Accuracy :", format(accuracy_score(y_test, y_pred),".2f"))
print("Precision:", format(precision_score(y_test, y_pred),".2f"))
print("Recall   :", format(recall_score(y_test, y_pred),".2f"))
print("F1 Score :", format(f1_score(y_test, y_pred),".2f"))

# ============ Confusion Matrix ==============

baseline_cm = confusion_matrix(y_test, baseline_pred)
tuned_cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrixs")

print("Baseline Confusion Matrix : ")
print(baseline_cm)
print("Hyperparameter Confusion Matrix : ")
print(tuned_cm)


# ================ Best Results in Grid Search CV =============
print("Best Parameters:", grid_search.best_params_)
print("Best Cross Validation Score:", grid_search.best_score_)


#================== Baseline VS Hyperparameter model ================

print("Baseline Model")
print(classification_report(y_test, baseline_pred)) 

print("Tuned Model")
print(classification_report(y_test, y_pred))

# ================== Graphical Representation =================



plt.figure(figsize=(10, 4))

# Baseline
plt.subplot(1, 2, 1)
sns.heatmap(
    baseline_cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)
plt.title("Baseline Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Tuned
plt.subplot(1, 2, 2)
sns.heatmap(
    tuned_cm,
    annot=True,
    fmt="d",
    cmap="Greens"
)
plt.title("Tuned Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.tight_layout()

plt.savefig("confusion_matrix.png")

plt.show()


with open("best_model.pkl", "wb") as file:
    pickle.dump(best_model, file)

with open("baseline_model.pkl", "wb") as file1:
    pickle.dump(baseline_model, file1)


print("Model saved successfully!")