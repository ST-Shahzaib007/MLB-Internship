import numpy as np

def main():
    print("==================================================")
    print("         Day 6: NumPy Practice Programs           ")
    print("==================================================")

    # 1. Create 1D and 2D arrays
    print("\n--- 1. Creating Arrays ---")
    arr_1d = np.array([10, 20, 30, 40, 50])
    arr_2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    
    print(f"1D Array (shape {arr_1d.shape}):")
    print(arr_1d)
    print(f"2D Array (shape {arr_2d.shape}):")
    print(arr_2d)

    # 2. Perform arithmetic operations on arrays
    print("\n--- 2. Arithmetic Operations ---")
    a = np.array([1, 2, 3, 4])
    b = np.array([5, 6, 7, 8])
    print(f"Array A: {a}")
    print(f"Array B: {b}")
    print(f"Addition (A + B):         {a + b}")
    print(f"Subtraction (B - A):      {b - a}")
    print(f"Multiplication (A * B):   {a * b}")
    print(f"Division (B / A):         {b / a}")
    print(f"Scalar Multiplication (A * 10): {a * 10}")

    # 3. Find the maximum, minimum, mean, and sum of an array
    print("\n--- 3. Statistical Functions ---")
    stats_arr = np.array([15, 22, 99, 45, 12, 88, 33])
    print(f"Target Array: {stats_arr}")
    print(f"Maximum: {np.max(stats_arr)}")
    print(f"Minimum: {np.min(stats_arr)}")
    print(f"Mean:    {np.mean(stats_arr):.2f}")
    print(f"Sum:     {np.sum(stats_arr)}")

    # 4. Reshape arrays into different dimensions
    print("\n--- 4. Array Reshaping ---")
    original_arr = np.arange(1, 13) # Array from 1 to 12
    print(f"Original 1D Array (shape {original_arr.shape}):")
    print(original_arr)
    
    reshaped_2d = original_arr.reshape(3, 4)
    print(f"Reshaped to 2D (3x4):")
    print(reshaped_2d)
    
    reshaped_3d = original_arr.reshape(2, 3, 2)
    print(f"Reshaped to 3D (2x3x2):")
    print(reshaped_3d)

    # 5. Slice and index arrays
    print("\n--- 5. Slicing and Indexing ---")
    # Slicing 1D
    print(f"1D Array: {arr_1d}")
    print(f"Element at index 2: {arr_1d[2]}")
    print(f"Slicing indices 1 to 4: {arr_1d[1:4]}")
    
    # Slicing 2D
    print("2D Array:")
    print(arr_2d)
    print(f"Element at row 1, col 2: {arr_2d[1, 2]}")
    print("First two rows:")
    print(arr_2d[:2, :])
    print("Last two columns:")
    print(arr_2d[:, 1:])

if __name__ == "__main__":
    main()
