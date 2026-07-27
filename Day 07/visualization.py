import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read cleaned data
df = pd.read_csv("cleaned_student_performance.csv")

# -----------------------------
# Bar Chart - Average Score
# -----------------------------
plt.figure(figsize=(10,5))
plt.bar(df["Name"], df["Average_Score"])
plt.title("Average Score per Student")
plt.xlabel("Students")
plt.ylabel("Average Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("bar_chart.png")
plt.close()

# -----------------------------
# Histogram
# -----------------------------
plt.figure(figsize=(7,5))
plt.hist(df["Average_Score"], bins=8)
plt.title("Average Score Distribution")
plt.xlabel("Average Score")
plt.ylabel("Number of Students")
plt.tight_layout()
plt.savefig("histogram.png")
plt.close()

# -----------------------------
# Scatter Plot
# -----------------------------
plt.figure(figsize=(7,5))
plt.scatter(df["Math"], df["Science"])
plt.xlabel("Math Marks")
plt.ylabel("Science Marks")
plt.title("Math vs Science")
plt.tight_layout()
plt.savefig("scatter_plot.png")
plt.close()

# -----------------------------
# Pie Chart
# -----------------------------
performance = df["Performance"].value_counts()

plt.figure(figsize=(6,6))
plt.pie(performance,
        labels=performance.index,
        autopct="%1.1f%%",
        startangle=90)

plt.title("Performance Categories")
plt.savefig("pie_chart.png")
plt.close()

# -----------------------------
# Box Plot
# -----------------------------
subjects=["Math","Science","English","Attendance"]

plt.figure(figsize=(8,5))
sns.boxplot(data=df[subjects])

plt.title("Marks Distribution")
plt.savefig("box_plot.png")
plt.close()

print("All Charts Created Successfully!")