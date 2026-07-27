import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

print("Everything Installed!")

df = pd.read_csv('student_performance.csv')
print(df.head())
print(df)