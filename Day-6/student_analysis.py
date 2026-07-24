import pandas as pd
import numpy as np

def main():
    print("==================================================")
    print("      Student Performance Analysis Project        ")
    print("==================================================")

    # 1. Load the dataset
    input_file = "student_data.csv"
    output_file = "student_analysis_results.csv"
    
    print(f"\n[Step 1] Loading dataset from: {input_file}")
    df = pd.read_csv(input_file)
    
    # 2. Display basic information about the dataset
    print("\n[Step 2] Basic Information:")
    print(f"Dataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("\nColumns and Data Types:")
    df.info()
    
    # Let's perform data cleaning: handle missing values
    print("\n[Data Cleaning] Checking and cleaning missing values...")
    missing_before = df.isnull().sum()
    print("Missing values before cleaning:")
    print(missing_before)
    
    # Calculate mean attendance to fill missing values
    mean_attendance = df["Attendance"].mean()
    df["Attendance"] = df["Attendance"].fillna(mean_attendance)
    print(f"Filled missing Attendance values with class mean: {mean_attendance:.2f}%")
    
    # 3. Calculate average marks for each subject
    print("\n[Step 3] Subject Averages:")
    avg_math = df["Math_Score"].mean()
    avg_science = df["Science_Score"].mean()
    avg_english = df["English_Score"].mean()
    print(f"Average Math Score:    {avg_math:.2f}")
    print(f"Average Science Score: {avg_science:.2f}")
    print(f"Average English Score: {avg_english:.2f}")
    
    # Add calculated columns for total and average score per student
    df["Total_Score"] = df["Math_Score"] + df["Science_Score"] + df["English_Score"]
    df["Average_Score"] = df["Total_Score"] / 3
    
    # Class overall average score
    class_average = df["Average_Score"].mean()
    print(f"Overall Class Average Score: {class_average:.2f}")

    # 4. Identify the top 5 performing students
    print("\n[Step 4] Top 5 Performing Students:")
    top_5 = df.nlargest(5, "Total_Score")
    print(top_5[["Student_ID", "Name", "Total_Score", "Average_Score"]].to_string(index=False))

    # 5. Find students scoring below the average
    print("\n[Step 5] Students Scoring Below Class Average:")
    below_avg = df[df["Average_Score"] < class_average]
    print(f"There are {len(below_avg)} students scoring below the overall class average of {class_average:.2f}:")
    print(below_avg[["Student_ID", "Name", "Total_Score", "Average_Score"]].to_string(index=False))

    # 6. Display the total number of students
    total_students = len(df)
    print(f"\n[Step 6] Total number of students: {total_students}")

    # 7. Save the cleaned or analyzed dataset as a new CSV file
    print(f"\n[Step 7] Saving analyzed and cleaned dataset to: {output_file}")
    # Round float columns for neatness
    df["Attendance"] = df["Attendance"].round(2)
    df["Average_Score"] = df["Average_Score"].round(2)
    df.to_csv(output_file, index=False)
    print("Dataset saved successfully!")

if __name__ == "__main__":
    main()
