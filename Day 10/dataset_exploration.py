import pandas as pd 
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df = pd.DataFrame(cancer.data, columns = cancer.feature_names)

df["Target"] = cancer.target


print(df.head())

print("Print Information", df.info())
print()
print("Satistical Sumary :", df.describe())

print("\nTarget Class Distribution:")
print(df["Target"].value_counts())