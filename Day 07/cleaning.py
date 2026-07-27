import pandas as pd

# Load CSV file
df = pd.read_csv("student_performance.csv")

print("========== ORIGINAL DATA ==========")
print(df.head())

# ---------------------------------
# Check Missing Values
# ---------------------------------
print("\nMissing Values:")
print(df.isnull().sum())

# Fill missing values in numeric columns with mean
numeric_columns = df.select_dtypes(include="number").columns
df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())

# Fill missing values in text columns with "Unknown"
text_columns = df.select_dtypes(include="object").columns
df[text_columns] = df[text_columns].fillna("Unknown")

# ---------------------------------
# Remove Duplicate Rows
# ---------------------------------
print("\nDuplicate Rows:", df.duplicated().sum())

df.drop_duplicates(inplace=True)

# ---------------------------------
# Rename Columns
# ---------------------------------
df.rename(columns={
    "Machine Learning": "Machine_Learning"
}, inplace=True)

# ---------------------------------
# Convert Data Types
# ---------------------------------
df["Math"] = df["Math"].astype(float)
df["Science"] = df["Science"].astype(float)
df["English"] = df["English"].astype(float)
df["Attendance"] = df["Attendance"].astype(float)

# ---------------------------------
# Create Average Score
# ---------------------------------
df["Average_Score"] = (
    df["Math"] +
    df["Science"] +
    df["English"]
) / 3

# ---------------------------------
# Create Performance Column
# ---------------------------------
def performance(avg):
    if avg >= 90:
        return "Excellent"
    elif avg >= 80:
        return "Good"
    elif avg >= 70:
        return "Average"
    else:
        return "Needs Improvement"

df["Performance"] = df["Average_Score"].apply(performance)

# ---------------------------------
# Sort Students by Average Score
# ---------------------------------
df = df.sort_values(by="Average_Score", ascending=False)

# ---------------------------------
# Save Cleaned File
# ---------------------------------
df.to_csv("cleaned_student_performance.csv", index=False)

print("\n========== CLEANED DATA ==========")
print(df.head())

print("\nCleaned file saved as cleaned_student_performance.csv")