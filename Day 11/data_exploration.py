# =========================== Libraries Required ====================
from sklearn.datasets import load_iris
import pandas as pd

# ================================= IRIS data loading =============================
iris = load_iris()

df = pd.DataFrame(iris.data,
                  columns = iris.feature_names)
print("Head")
print(df.head())
print("Simple DF")
print(df)
print("Columns")        # +++++++++++  Dataset manuplutation
print(df.columns)
print("Info")
print(df.info())
print("Shape")
print(df.shape)