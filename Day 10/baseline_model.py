import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (

accuracy_score,
f1_score,
precision_score,
recall_score,
confusion_matrix,
classification_report

)

cancer = load_breast_cancer()


x = cancer.data
y = cancer.target


x_train, x_test, y_train, y_test = train_test_split(

    x,
    y,
    test_size =0.2,
    random_state = 42
)

model = LogisticRegression(max_iter=10000)


model.fit(x_train,y_train)

y_predict = model.predict(x_test)

# Evaluation
print("Accuracy :", format(accuracy_score(y_test, y_predict),".2f"))
print("Precision:", format(precision_score(y_test, y_predict),".2f"))
print("Recall   :", format(recall_score(y_test, y_predict), ".2f"))
print("F1 Score :", format(f1_score(y_test, y_predict),".2f"))  

cm = confusion_matrix(y_test,y_predict)

print("Confusion matrix : ")
print(cm)


classification = classification_report(
    y_test,
    y_predict,
    target_names= cancer.target_names

)

print("Claasification : ")
print(classification)