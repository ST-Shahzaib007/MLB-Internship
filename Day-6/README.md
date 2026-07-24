# Day 6: Python for Data Science

Welcome to Day 6 of the MLB Internship roadmap! Today focuses on the fundamentals of **NumPy** and **Pandas**, two of the most widely used libraries in Python for data scientific computation and analysis.

---

## 📖 Table of Contents
1. [What I Learned: NumPy](#-what-i-learned-numpy)
2. [What I Learned: Pandas](#-what-i-learned-pandas)
3. [Mini Project: Student Performance Analysis](#-mini-project-student-performance-analysis)
4. [💡 Key Insights from the Dataset](#-key-insights-from-the-dataset)
5. [⚠️ Challenges & Solutions](#-challenges--solutions)
6. [📂 Folder Structure](#-folder-structure)

---

## 🔢 What I Learned: NumPy

NumPy (Numerical Python) is the foundation of scientific and numerical computing in Python. It provides high-performance multidimensional array objects (`ndarray`) and tools for working with them. Key concepts learned include:

- **Creating Arrays**: Initiating 1D and 2D arrays using `np.array()`, and auto-populating arrays with utilities like `np.arange()`.
- **Arithmetic & Element-wise Operations**: Performing mathematical computations (`+`, `-`, `*`, `/`) directly on arrays without loops (vectorized operations), which is computationally efficient.
- **Statistical Summarization**: Finding key statistics using `np.max()`, `np.min()`, `np.mean()`, and `np.sum()`.
- **Reshaping Dimensions**: Altering array dimensions (e.g., transforming a 1D array of 12 elements into a 2D 3x4 array or a 3D 2x3x2 array) using `.reshape()`.
- **Slicing and Indexing**: Accessing subparts of 1D and 2D arrays using ranges (e.g., `array[start:end]` and `array[row_start:row_end, col_start:col_end]`).

---

## 🐼 What I Learned: Pandas

Pandas is a powerful, flexible library built for data manipulation and analysis. It provides two main data structures: `Series` (1D labeled array) and `DataFrame` (2D tabular structure). Key operations learned include:

- **Series & DataFrames**: Constructing datasets from dictionary payloads or single columns manually.
- **Reading Data**: Loading external datasets into DataFrames using `pd.read_csv()`.
- **Data Exploration**:
  - `df.head()` / `df.tail()`: Reviewing first/last rows.
  - `df.info()`: Inspecting data types, non-null counts, and memory footprint.
  - `df.describe()`: Displaying statistical properties (mean, std, min, percentiles, max) for numeric columns.
- **Data Cleaning**: Counting null rows with `df.isnull().sum()` and filling missing values with mean substitution using `.fillna()`.
- **Data Filtering**: Filtering rows based on specific conditions (e.g. finding students scoring above 80 in mathematics).

---

## 🎓 Mini Project: Student Performance Analysis

Using the dataset `student_data.csv` containing performance metrics of 20 students, the `student_analysis.py` script executes the following workflow:
1. **Loads the Student Records** using Pandas.
2. **Performs Data Cleaning** by substituting two missing `Attendance` values with the class mean attendance rate of **89.23%**.
3. **Calculates Subject Averages** across the entire class:
   - **Math Average**: `70.25`
   - **Science Average**: `72.95`
   - **English Average**: `73.15`
4. **Calculates Personal Metrics**: Computes `Total_Score` and `Average_Score` for each student.
5. **Identifies Top 5 Performing Students** based on their total scores.
6. **Identifies Below-Average Students** whose individual `Average_Score` is less than the class overall average score of `72.12`.
7. **Saves Results**: Writes the cleaned and analyzed table to `student_analysis_results.csv`.

---

## 💡 Key Insights from the Dataset

- **Overall Class Performance**: The overall average score of the class is **72.12 / 100**, indicating a solid passing performance.
- **Subject-wise Strengths**: Students performed best on average in **English** (73.15) and worst in **Math** (70.25). This shows that Math is a potential area for curriculum review or student tutoring.
- **Outstanding Performers**:
  - **Grace** (S007) is the top performer with a near-perfect score of **96.67%** (Total: 290).
  - **Sofia** (S019) followed closely at **92.67%** (Total: 278).
- **Academic Support Needs**: Out of 20 students, **9 students (45%)** scored below the overall class average of 72.12. Notable students requiring urgent help include **Kate** (S011) with a total score of 125 (Avg: 41.67) and **Ian** (S009) with a total score of 127 (Avg: 42.33).
- **Attendance vs. Performance**: There appears to be a strong positive correlation between attendance rate and score. Top performers Grace (99.0%), Sofia (96.5%), and Charlie (98.2%) have excellent attendance. Conversely, low-scoring students like Kate (70.5%) and David (75.0%) have attendance rates below 80%.

---

## ⚠️ Challenges & Solutions

1. **Handling Missing Values in Real-World Contexts**:
   - *Challenge*: The attendance column contained blank entries (missing values). Leaving them as `NaN` results in gaps during correlation and average calculations.
   - *Solution*: Leveraged Pandas' `df['Attendance'].fillna(df['Attendance'].mean(), inplace=True)` to replace null elements with the class mean attendance rate, maintaining statistical balance.
2. **Dimension Constraints during Array Reshaping**:
   - *Challenge*: A common error in NumPy occurs when attempting to reshape an array into dimensions that do not match the total number of elements (e.g. reshaping an array of 12 elements into a 5x2 grid).
   - *Solution*: Learned to compute dimensions beforehand. For an array of size $N$, the dimensions $d_1 \times d_2 \times \dots \times d_k$ must satisfy $\prod d_i = N$.

---

## 📂 Folder Structure

```
Day-6/
├── numpy_practice.py             # NumPy practice script
├── pandas_practice.py            # Pandas practice script
├── student_analysis.py           # Student performance analysis project
├── student_data.csv              # Original student dataset
├── student_analysis_results.csv  # Cleaned and processed dataset
└── README.md                     # Documentation
```
