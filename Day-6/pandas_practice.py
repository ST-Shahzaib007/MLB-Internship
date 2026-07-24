import pandas as pd

def main():
    print("==================================================")
    print("         Day 6: Pandas Practice Programs          ")
    print("==================================================")

    # 1. Creating Series and DataFrames manually
    print("\n--- 1. Creating Pandas Series and DataFrames Manually ---")
    # Series
    fruits = pd.Series(["Apple", "Banana", "Cherry"], name="Fruit")
    print("Pandas Series:")
    print(fruits)
    print()

    # DataFrame
    simple_df = pd.DataFrame({
        "Product": ["Laptop", "Mouse", "Keyboard"],
        "Price": [999.99, 25.50, 75.00],
        "Stock": [10, 150, 45]
    })
    print("Pandas DataFrame:")
    print(simple_df)

    # Path to our dataset
    csv_file = "student_data.csv"

    # 2. Loading the dataset
    print(f"\n--- 2. Loading Dataset ({csv_file}) ---")
    df = pd.read_csv(csv_file)
    print("Dataset loaded successfully!")

    # 3. Displaying the first and last five rows
    print("\n--- 3. Displaying Head and Tail ---")
    print("First 5 rows (head):")
    print(df.head())
    print("\nLast 5 rows (tail):")
    print(df.tail())

    # 4. Displaying dataset information
    print("\n--- 4. Dataset Information ---")
    df.info()

    # 5. Finding missing values
    print("\n--- 5. Finding Missing Values ---")
    missing_vals = df.isnull().sum()
    print("Number of missing values per column:")
    print(missing_vals)
    
    # Check if there are any rows with missing values
    missing_rows = df[df.isnull().any(axis=1)]
    print("\nRows with missing values:")
    print(missing_rows)

    # 6. Filtering data based on a condition
    print("\n--- 6. Filtering Data (Math_Score > 80) ---")
    high_math = df[df["Math_Score"] > 80]
    print(f"Students scoring > 80 in Math (Total: {len(high_math)}):")
    print(high_math[["Student_ID", "Name", "Math_Score"]])

    # 7. Summary statistics
    print("\n--- 7. Summary Statistics ---")
    print(df.describe())

if __name__ == "__main__":
    main()
