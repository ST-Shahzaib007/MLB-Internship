import pandas as pd

df = pd.read_csv("cleaned_student_performance.csv")

subjects = ["Math","Science","English","Attendance"]

print("="*50)
print("      STUDENT PERFORMANCE DASHBOARD")
print("="*50)

# Total Students
print("\nTotal Students:", len(df))

# Average score of every subject
print("\nAverage Score Per Subject")
print(df[subjects].mean())

# Top 5 Students
print("\nTop 5 Students")
top5 = df.sort_values(by="Average_Score", ascending=False)
print(top5[["Name","Average_Score"]].head())

# Students Needing Improvement
print("\nStudents Needing Improvement")
need = df[df["Performance"]=="Needs Improvement"]
print(need[["Name","Average_Score","Performance"]])

# Highest Average Subject
print("\nSubject With Highest Average")
print(df[subjects].mean().idxmax())

print("\nDashboard Completed Successfully!")