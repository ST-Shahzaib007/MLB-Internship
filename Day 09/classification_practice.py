from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import(
classification_report,
f1_score,
accuracy_score,
precision_score,
recall_score,
classification_report,
confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns

iris = load_iris()

x = iris.data
y= iris.target
model = LogisticRegression(max_iter =500)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size = 0.2,
    random_state =42
)

model.fit(x_train,y_train)

print("Model Trained Successfully... !")
print()

y_predict = model.predict(x_test)

print("Prediction : ")
print(y_predict)

print("Actual : ")
print(y_test)

accuracy = accuracy_score(y_test, y_predict)
f1 = f1_score(y_test,y_predict,
              average ="weighted"
              )
recall = recall_score(y_test,y_predict,
              average ="weighted"
              )
precision = precision_score(y_test,y_predict,
              average ="weighted"
              )
classification = classification_report(y_test,
                                       y_predict,
                                       target_names = iris.target_names)

conf_matrix = confusion_matrix(y_test, y_predict)
print("Accuracy : ", accuracy)
print("F1 Score : ", f1)    
print("Recall : ", recall)
print("Precision : ", precision)
print("Classification Report : ")
print(classification)
print("Confusion Matrix : ")
print(conf_matrix)



print("\nSample Predictions:\n")

for i in range(10):
    print(
        f"Sample {i+1}: "
        f"Predicted = {iris.target_names[y_predict[i]]}, "
        f"Actual = {iris.target_names[y_test[i]]}"
    )



# Plotting the confusion matrix

plt.figure(figsize=(6,5))
sns.heatmap(conf_matrix, annot = True, fmt= "d",  cmap ="Greens", 
            xticklabels = iris.target_names, 
            yticklabels = iris.target_names)
plt.xlabel ("Predicted Label")
plt.ylabel ("Actual Label")
plt.title("Confusion Matrix")
plt.show()
print("Confusion Matrix plotted successfully... !")

