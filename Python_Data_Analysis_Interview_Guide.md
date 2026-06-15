# 🐍 Python for Data Analysis — Complete Interview Guide
> A comprehensive reference covering NumPy, Pandas, Matplotlib, Seaborn, Data Cleaning, Statistics, Machine Learning, NLP, Deep Learning, and Big Data — with detailed answers and examples for interviews.

---

## Table of Contents

1. [NumPy](#1-numpy)
2. [Pandas](#2-pandas)
3. [Matplotlib & Seaborn](#3-matplotlib--seaborn)
4. [Data Cleaning & Preprocessing](#4-data-cleaning--preprocessing)
5. [Statistical Analysis with Python](#5-statistical-analysis-with-python)
6. [Machine Learning with Scikit-Learn](#6-machine-learning-with-scikit-learn)
7. [Time Series Analysis](#7-time-series-analysis)
8. [Web Scraping with BeautifulSoup & Requests](#8-web-scraping-with-beautifulsoup--requests)
9. [SQL for Data Analysis (Python Integration)](#9-sql-for-data-analysis-python-integration)
10. [Advanced Data Visualization](#10-advanced-data-visualization)
11. [Natural Language Processing (NLP)](#11-natural-language-processing-nlp)
12. [Deep Learning with TensorFlow](#12-deep-learning-with-tensorflow)
13. [Transfer Learning with Pre-trained Models](#13-transfer-learning-with-pre-trained-models)
14. [Big Data Processing with Apache Spark](#14-big-data-processing-with-apache-spark)

---

## 1. NumPy

### What is NumPy and why is it used?

**NumPy** (Numerical Python) is the foundational package for scientific computing in Python. It provides:
- A powerful **N-dimensional array** object (`ndarray`)
- Fast **mathematical operations** on arrays (vectorized, no loops needed)
- Tools for integrating C/C++ and Fortran code
- Linear algebra, Fourier transforms, and random number capabilities

NumPy arrays are significantly faster than Python lists because they store data in contiguous memory and avoid the overhead of Python's dynamic typing.

```python
import numpy as np

# Python list vs NumPy array speed difference
python_list = list(range(1_000_000))
numpy_array = np.arange(1_000_000)

# NumPy is typically 10–100x faster for numerical operations
result = numpy_array * 2   # vectorized — no for loop needed
```

---

### Array Creation

**Interview Tip:** Know the difference between these common creation functions.

```python
import numpy as np

# From a Python list
arr = np.array([1, 2, 3, 4, 5])
print(arr)          # [1 2 3 4 5]
print(arr.dtype)    # int64

# 2D array (matrix)
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
print(matrix.shape)  # (2, 3) — 2 rows, 3 columns

# Zeros and ones
zeros  = np.zeros((3, 4))      # 3x4 matrix of 0.0
ones   = np.ones((2, 3))       # 2x3 matrix of 1.0
eye    = np.eye(3)             # 3x3 identity matrix

# Ranges
arr_range  = np.arange(0, 10, 2)        # [0 2 4 6 8]
arr_linspace = np.linspace(0, 1, 5)     # [0.   0.25 0.5  0.75 1.  ] — evenly spaced

# Random arrays
rand_uniform = np.random.rand(3, 3)         # uniform [0, 1)
rand_normal  = np.random.randn(3, 3)        # standard normal distribution
rand_int     = np.random.randint(1, 10, (3, 3))  # random integers 1–9

# Full array with a constant value
full_arr = np.full((2, 3), 7)   # [[7, 7, 7], [7, 7, 7]]
```

---

### Array Manipulation

```python
arr = np.array([1, 2, 3, 4, 5, 6])

# Reshape — change shape without changing data
matrix = arr.reshape(2, 3)    # [[1,2,3],[4,5,6]]
flat   = matrix.flatten()     # Back to [1,2,3,4,5,6]

# Transpose
print(matrix.T)               # Swap rows and columns

# Stack arrays
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

horizontal = np.hstack([a, b])      # [1 2 3 4 5 6]
vertical   = np.vstack([a, b])      # [[1,2,3],[4,5,6]]

# Element-wise operations
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(arr1 + arr2)    # [5 7 9]
print(arr1 * arr2)    # [4 10 18]
print(arr1 ** 2)      # [1 4 9]
```

---

### Mathematical Operations on Arrays

```python
arr = np.array([1, 2, 3, 4, 5])

# Basic stats
print(np.mean(arr))    # 3.0
print(np.median(arr))  # 3.0
print(np.std(arr))     # 1.414...
print(np.var(arr))     # 2.0
print(np.sum(arr))     # 15
print(np.min(arr))     # 1
print(np.max(arr))     # 5

# Math functions
print(np.sqrt(arr))    # [1.  1.41 1.73 2.   2.23]
print(np.log(arr))     # natural log
print(np.exp(arr))     # e^x for each element

# Axis-based operations on 2D arrays
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

print(np.sum(matrix, axis=0))  # Sum of each COLUMN: [5, 7, 9]
print(np.sum(matrix, axis=1))  # Sum of each ROW:    [6, 15]
```

---

### Broadcasting

Broadcasting allows NumPy to perform operations on arrays of **different shapes** by automatically expanding the smaller array to match the larger one.

```python
arr = np.array([1, 2, 3])

# Scalar broadcast — multiplies every element by 2
result = arr * 2           # [2 4 6]

# Add a 1D array to each row of a 2D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])
row    = np.array([10, 20, 30])

result = matrix + row
# [[11, 22, 33],
#  [14, 25, 36]]
```

**Rule:** Arrays are compatible for broadcasting if their dimensions match from the trailing end (right side), or one of them is 1.

---

### Indexing and Slicing

```python
arr = np.array([10, 20, 30, 40, 50])

# Basic indexing
print(arr[0])     # 10 (first element)
print(arr[-1])    # 50 (last element)

# Slicing [start:stop:step]
print(arr[1:4])   # [20 30 40] — index 1 to 3 (stop is exclusive)
print(arr[::2])   # [10 30 50] — every other element
print(arr[::-1])  # [50 40 30 20 10] — reversed

# 2D array indexing
matrix = np.array([[1, 2, 3],
                   [4, 5, 6],
                   [7, 8, 9]])

print(matrix[0, 1])       # 2 — row 0, col 1
print(matrix[1, :])       # [4 5 6] — entire row 1
print(matrix[:, 2])       # [3 6 9] — entire column 2
print(matrix[0:2, 1:3])   # [[2,3],[5,6]] — submatrix

# Boolean indexing (very powerful for filtering)
arr = np.array([1, 2, 3, 4, 5, 6])
print(arr[arr > 3])        # [4 5 6] — elements greater than 3
print(arr[(arr > 2) & (arr < 5)])  # [3 4]
```

---

### Common Interview Questions on NumPy

**Q: What is the difference between `copy()` and a view in NumPy?**

```python
arr = np.array([1, 2, 3, 4, 5])

# View — shares memory with original
view = arr[1:4]
view[0] = 99
print(arr)    # [1 99 3 4 5] — original is modified!

# Copy — independent memory
copy = arr[1:4].copy()
copy[0] = 0
print(arr)    # [1 99 3 4 5] — original unchanged
```

**Q: What is `np.where()` and how is it used?**

```python
arr = np.array([1, -2, 3, -4, 5])

# Replace negatives with 0
result = np.where(arr > 0, arr, 0)
print(result)  # [1 0 3 0 5]
```

---

## 2. Pandas

### What is Pandas and why is it important?

**Pandas** is the go-to Python library for data manipulation and analysis. It provides two main data structures:
- **Series** — one-dimensional labeled array (like a single column)
- **DataFrame** — two-dimensional labeled table (like a spreadsheet or SQL table)

Pandas makes it easy to load, clean, filter, aggregate, and visualize data from CSV, Excel, JSON, SQL, and many other formats.

---

### Series vs DataFrame

```python
import pandas as pd
import numpy as np

# Series — 1D labeled array
s = pd.Series([10, 20, 30, 40], index=['a', 'b', 'c', 'd'])
print(s['b'])     # 20
print(s[s > 15])  # filter: b=20, c=30, d=40

# DataFrame — 2D table
df = pd.DataFrame({
    'Name':   ['Alice', 'Bob', 'Charlie', 'Diana'],
    'Age':    [25, 30, 35, 28],
    'City':   ['New York', 'San Francisco', 'Los Angeles', 'Chicago'],
    'Salary': [70000, 80000, 90000, 75000]
})

print(df.shape)        # (4, 4)
print(df.dtypes)       # data types per column
print(df.head(2))      # first 2 rows
print(df.tail(2))      # last 2 rows
print(df.info())       # memory usage, non-null counts
print(df.describe())   # count, mean, std, min, max, percentiles
```

---

### Selecting Data

```python
# Select a single column (returns Series)
print(df['Name'])

# Select multiple columns (returns DataFrame)
print(df[['Name', 'Salary']])

# Select rows by index label — loc
print(df.loc[0])            # row at index label 0
print(df.loc[0:2, 'Name':'Age'])  # rows 0-2, columns Name to Age

# Select rows by integer position — iloc
print(df.iloc[0])           # first row
print(df.iloc[0:2, 0:2])    # first 2 rows, first 2 columns

# Conditional filtering
adults    = df[df['Age'] > 25]
high_earn = df[(df['Salary'] > 70000) & (df['City'] == 'San Francisco')]
```

---

### Data Cleaning and Manipulation

**Handling Missing Data:**

```python
# Create DataFrame with NaNs
df = pd.DataFrame({
    'Name':   ['Alice', 'Bob', None, 'Diana'],
    'Age':    [25, np.nan, 35, 28],
    'Salary': [70000, 80000, 90000, np.nan]
})

# Detect missing values
print(df.isnull())            # True where value is NaN
print(df.isnull().sum())      # Count of NaN per column
print(df.isnull().sum() / len(df) * 100)  # % missing per column

# Drop rows with ANY missing value
df_dropped = df.dropna()

# Drop rows where ALL values are missing
df_dropped_all = df.dropna(how='all')

# Drop columns with missing values
df_dropped_cols = df.dropna(axis=1)

# Fill missing values
df['Age'].fillna(df['Age'].mean(), inplace=True)       # Fill with mean
df['Name'].fillna('Unknown', inplace=True)             # Fill with constant
df['Salary'].fillna(method='ffill', inplace=True)      # Forward fill
df['Salary'].fillna(method='bfill', inplace=True)      # Backward fill
```

**Adding, Renaming, and Removing Columns:**

```python
# Add a new column
df['Bonus'] = df['Salary'] * 0.1

# Computed column
df['Total_Comp'] = df['Salary'] + df['Bonus']

# Rename columns
df.rename(columns={'Name': 'Employee', 'Age': 'Years'}, inplace=True)

# Remove a column
df.drop('Bonus', axis=1, inplace=True)

# Remove multiple columns
df.drop(['Bonus', 'Total_Comp'], axis=1, inplace=True)

# Remove rows by index
df.drop([0, 2], axis=0, inplace=True)
```

**Filtering and Sorting:**

```python
# Filter with multiple conditions
filtered = df[(df['Age'] > 25) & (df['Salary'] > 70000)]

# isin — check membership
cities = ['New York', 'Chicago']
subset = df[df['City'].isin(cities)]

# Sort by one or multiple columns
df_sorted = df.sort_values('Salary', ascending=False)
df_sorted = df.sort_values(['Age', 'Salary'], ascending=[True, False])

# Reset index after filtering/sorting
df_sorted.reset_index(drop=True, inplace=True)
```

---

### Grouping and Aggregation

```python
df = pd.DataFrame({
    'Department': ['IT', 'IT', 'HR', 'HR', 'Finance'],
    'Employee':   ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Salary':     [80000, 90000, 60000, 65000, 75000],
    'Experience': [5, 7, 3, 4, 6]
})

# GroupBy — group by one column
grouped = df.groupby('Department')

# Single aggregation
print(grouped['Salary'].mean())    # average salary per department
print(grouped['Salary'].sum())     # total salary per department
print(grouped['Salary'].count())   # headcount per department

# Multiple aggregations at once
summary = grouped['Salary'].agg(['mean', 'sum', 'min', 'max', 'count'])

# Different aggregations per column
summary = grouped.agg({
    'Salary':     ['mean', 'max'],
    'Experience': 'mean'
})

# GroupBy on multiple columns
multi = df.groupby(['Department', 'Experience'])['Salary'].mean()

# Apply a custom function
def salary_range(x):
    return x.max() - x.min()

grouped['Salary'].apply(salary_range)
```

---

### Apply, Map, and Lambda

```python
# apply — apply a function to rows or columns
df['Salary_K'] = df['Salary'].apply(lambda x: x / 1000)

# apply across rows (axis=1)
df['Category'] = df.apply(
    lambda row: 'Senior' if row['Experience'] > 5 else 'Junior', axis=1
)

# map — element-wise transformation on a Series
dept_map = {'IT': 'Technology', 'HR': 'Human Resources', 'Finance': 'Finance'}
df['Dept_Full'] = df['Department'].map(dept_map)
```

---

### Merging and Joining DataFrames

```python
employees = pd.DataFrame({
    'emp_id': [1, 2, 3, 4],
    'name':   ['Alice', 'Bob', 'Charlie', 'Diana'],
    'dept_id': [10, 20, 10, 30]
})

departments = pd.DataFrame({
    'dept_id':   [10, 20, 40],
    'dept_name': ['IT', 'HR', 'Finance']
})

# INNER JOIN — only matching rows
inner = pd.merge(employees, departments, on='dept_id', how='inner')

# LEFT JOIN — all employees, NaN if no dept
left  = pd.merge(employees, departments, on='dept_id', how='left')

# RIGHT JOIN — all departments, NaN if no employees
right = pd.merge(employees, departments, on='dept_id', how='right')

# OUTER JOIN — all rows from both
outer = pd.merge(employees, departments, on='dept_id', how='outer')

# Concatenate DataFrames (stack vertically)
combined = pd.concat([df1, df2], ignore_index=True)

# Concatenate horizontally
side_by_side = pd.concat([df1, df2], axis=1)
```

---

### Pivot Tables

```python
df = pd.DataFrame({
    'Month':    ['Jan', 'Jan', 'Feb', 'Feb'],
    'Category': ['A', 'B', 'A', 'B'],
    'Sales':    [100, 200, 150, 250]
})

pivot = df.pivot_table(
    values='Sales',
    index='Month',
    columns='Category',
    aggfunc='sum',
    fill_value=0
)
```

---

### Common Interview Questions on Pandas

**Q: What is the difference between `loc` and `iloc`?**

```python
# loc — label-based: uses row/column NAMES
df.loc[0, 'Name']       # row label 0, column 'Name'
df.loc[0:3, 'A':'C']    # inclusive on both ends

# iloc — integer-based: uses row/column POSITIONS
df.iloc[0, 1]           # row 0, column 1
df.iloc[0:3, 0:2]       # exclusive end (like Python slicing)
```

**Q: How do you handle duplicate rows in Pandas?**

```python
print(df.duplicated().sum())          # count duplicates
df_clean = df.drop_duplicates()       # remove all duplicates
df_clean = df.drop_duplicates(subset=['Name', 'Age'])  # based on specific columns
df_clean = df.drop_duplicates(keep='last')             # keep last occurrence
```

**Q: What is `value_counts()` and when do you use it?**

```python
# Frequency count of unique values
print(df['Department'].value_counts())
print(df['Department'].value_counts(normalize=True))  # proportions (%)
```

---

## 3. Matplotlib & Seaborn

### What is Matplotlib and why is it used?

**Matplotlib** is Python's foundational plotting library. It gives you fine-grained control over every aspect of a figure. **Seaborn** is built on top of Matplotlib and provides a higher-level interface with beautiful defaults, especially for statistical visualizations.

---

### Basic Plots with Matplotlib

```python
import matplotlib.pyplot as plt
import numpy as np

x = [1, 2, 3, 4, 5]
y = [2, 4, 6, 8, 10]

# Line plot — for trends over time
plt.figure(figsize=(8, 5))
plt.plot(x, y, color='blue', linewidth=2, linestyle='--', marker='o', label='Growth')
plt.xlabel('X-axis Label', fontsize=12)
plt.ylabel('Y-axis Label', fontsize=12)
plt.title('Line Plot Example', fontsize=16)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Bar chart — for comparing categories
categories = ['A', 'B', 'C', 'D']
values     = [30, 45, 25, 60]

plt.bar(categories, values, color=['steelblue', 'salmon', 'green', 'purple'])
plt.title('Bar Chart')
plt.show()

# Scatter plot — for relationships between two variables
x = np.random.randn(100)
y = x * 2 + np.random.randn(100)

plt.scatter(x, y, alpha=0.6, color='coral', edgecolors='k')
plt.title('Scatter Plot')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Histogram — for distribution of a single variable
data = np.random.randn(1000)
plt.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
plt.title('Histogram')
plt.show()

# Pie chart — for proportions
sizes  = [35, 30, 20, 15]
labels = ['Category A', 'Category B', 'Category C', 'Category D']
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart')
plt.show()
```

---

### Subplots — Multiple Plots in One Figure

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # 1 row, 2 columns

axes[0].plot(x, y)
axes[0].set_title('Line Plot')

axes[1].bar(categories, values)
axes[1].set_title('Bar Chart')

plt.tight_layout()
plt.show()
```

---

### Seaborn for Statistical Visualization

```python
import seaborn as sns
import pandas as pd

# Set a theme
sns.set_theme(style='whitegrid', palette='muted')

# Load a sample dataset
df = sns.load_dataset('tips')

# Distribution plot
sns.histplot(df['total_bill'], kde=True, bins=20)
plt.title('Bill Distribution')
plt.show()

# Box plot — shows median, quartiles, outliers
sns.boxplot(x='day', y='total_bill', data=df)
plt.title('Bills by Day')
plt.show()

# Violin plot — combines box plot + KDE
sns.violinplot(x='day', y='total_bill', data=df)
plt.show()

# Bar plot with error bars (confidence interval)
sns.barplot(x='day', y='total_bill', data=df, ci=95)
plt.show()

# Scatter plot with color by category
sns.scatterplot(x='total_bill', y='tip', hue='sex', size='size', data=df)
plt.show()

# Regression plot — scatter + trend line
sns.regplot(x='total_bill', y='tip', data=df)
plt.show()

# Heatmap — for correlation matrices
corr = df.select_dtypes(include='number').corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Heatmap')
plt.show()

# Pair plot — scatter matrix for all numeric columns
sns.pairplot(df, hue='sex')
plt.show()

# Count plot — like bar chart for categorical variables
sns.countplot(x='day', data=df, palette='Set2')
plt.show()
```

---

### Data Visualization Best Practices

| Scenario | Recommended Chart |
|---------|------------------|
| Trend over time | Line plot |
| Compare categories | Bar chart |
| Distribution of one variable | Histogram, Box plot, Violin |
| Relationship between two variables | Scatter plot, Regression plot |
| Proportion of a whole | Pie chart, Stacked bar |
| Correlation between many variables | Heatmap, Pair plot |
| Outlier detection | Box plot |
| Categorical vs numeric | Box plot, Violin plot, Bar plot |

**Key Principles:**
- Label axes and add a title
- Use color meaningfully — not just decoratively
- Avoid 3D charts (they distort perception)
- Minimize clutter — remove unnecessary gridlines, borders
- Choose the right chart for the story you want to tell

---

## 4. Data Cleaning & Preprocessing

### Why is data cleaning important?

Real-world data is messy — it contains missing values, duplicates, inconsistent formats, outliers, and wrong data types. Cleaning accounts for 60–80% of a data analyst's time. Models trained on dirty data produce poor results ("garbage in, garbage out").

---

### Handling Missing Data

```python
import pandas as pd
import numpy as np

df = pd.read_csv('data.csv')

# Step 1: Understand the extent of missing data
print(df.isnull().sum())
print(df.isnull().sum() / len(df) * 100)  # % missing

# Step 2: Decide strategy
# — Drop if missing > 40–50% of a column
# — Impute (fill) if small percentage

# Drop columns with too many missing values
df.dropna(thresh=len(df) * 0.6, axis=1, inplace=True)  # keep cols with 60%+ non-null

# Fill numeric with mean or median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill categorical with mode
df['City'].fillna(df['City'].mode()[0], inplace=True)

# Forward fill / backward fill (for time series)
df['Price'].fillna(method='ffill', inplace=True)
```

---

### Removing Duplicates

```python
# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# Drop all duplicates (keep first occurrence)
df = df.drop_duplicates()

# Drop duplicates based on specific columns
df = df.drop_duplicates(subset=['customer_id', 'order_date'])

# Keep last occurrence
df = df.drop_duplicates(keep='last')
```

---

### Data Type Conversion

```python
# Check types
print(df.dtypes)

# Convert types
df['Age']        = df['Age'].astype(int)
df['Price']      = df['Price'].astype(float)
df['Date']       = pd.to_datetime(df['Date'])
df['Category']   = df['Category'].astype('category')  # memory efficient
df['is_active']  = df['is_active'].astype(bool)

# String cleaning
df['Name'] = df['Name'].str.strip()       # remove whitespace
df['Name'] = df['Name'].str.lower()       # lowercase
df['Name'] = df['Name'].str.title()       # Title Case
df['Code'] = df['Code'].str.replace('-', '', regex=False)  # remove dashes
```

---

### Data Normalization and Scaling

Scaling ensures features with large ranges don't dominate those with small ranges — critical for algorithms like KNN, SVM, Gradient Descent.

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler

df_numeric = df[['Age', 'Salary', 'Experience']]

# Min-Max Scaling — rescales to [0, 1]
# Best when: you need bounded values and the distribution is NOT Gaussian
scaler = MinMaxScaler()
df_minmax = pd.DataFrame(
    scaler.fit_transform(df_numeric),
    columns=df_numeric.columns
)
# Formula: (x - min) / (max - min)

# Standardization (Z-score) — mean=0, std=1
# Best when: data follows a Gaussian distribution; required for PCA, linear models
scaler = StandardScaler()
df_standard = pd.DataFrame(
    scaler.fit_transform(df_numeric),
    columns=df_numeric.columns
)
# Formula: (x - mean) / std

# Robust Scaling — uses median and IQR, less sensitive to outliers
scaler = RobustScaler()
df_robust = pd.DataFrame(
    scaler.fit_transform(df_numeric),
    columns=df_numeric.columns
)
```

---

### Handling Categorical Data

Models require numeric input — categorical data must be encoded.

```python
# One-Hot Encoding — creates binary columns for each category
# Use when: the variable is NOMINAL (no order), e.g., Color, City
encoded = pd.get_dummies(df['City'], prefix='city')
df = pd.concat([df, encoded], axis=1)
df.drop('City', axis=1, inplace=True)

# Pandas built-in for entire DataFrame
df_encoded = pd.get_dummies(df, columns=['City', 'Department'])

# Label Encoding — converts categories to integers 0, 1, 2, ...
# Use when: the variable is ORDINAL (has order), e.g., Low/Medium/High
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
df['Department_Encoded'] = le.fit_transform(df['Department'])
# IT=1, HR=0, Finance=2 (alphabetical order — no inherent order)

# Ordinal Encoding (manual mapping for ordered categories)
order_map = {'Low': 0, 'Medium': 1, 'High': 2}
df['Priority'] = df['Priority'].map(order_map)
```

---

### Outlier Detection and Handling

```python
# IQR Method
Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = df[(df['Salary'] < lower_bound) | (df['Salary'] > upper_bound)]
print(f"Outliers found: {len(outliers)}")

# Remove outliers
df_clean = df[(df['Salary'] >= lower_bound) & (df['Salary'] <= upper_bound)]

# Cap outliers (winsorizing) instead of removing
df['Salary'] = df['Salary'].clip(lower=lower_bound, upper=upper_bound)

# Z-score method
from scipy.stats import zscore
z_scores = zscore(df['Salary'])
df_clean  = df[abs(z_scores) < 3]  # remove rows where z-score > 3
```

---

## 5. Statistical Analysis with Python

### Descriptive Statistics

Descriptive statistics summarize the main features of a dataset without drawing conclusions about a larger population.

```python
import pandas as pd
import numpy as np
from scipy import stats

df = pd.read_csv('employees.csv')

# Central tendency — where is the data centered?
mean_val   = df['Salary'].mean()
median_val = df['Salary'].median()
mode_val   = df['Salary'].mode()[0]

# Dispersion — how spread out is the data?
std_dev    = df['Salary'].std()
variance   = df['Salary'].var()
range_val  = df['Salary'].max() - df['Salary'].min()
iqr        = df['Salary'].quantile(0.75) - df['Salary'].quantile(0.25)

# Shape
skewness   = df['Salary'].skew()     # > 0 right-skewed, < 0 left-skewed
kurtosis   = df['Salary'].kurt()     # > 3 heavy tails (leptokurtic)

# Quick summary
print(df.describe(percentiles=[.25, .5, .75, .95]))
```

**Interview Tip:** Mean is sensitive to outliers; use **median** for skewed data. Use **mode** for categorical data.

---

### Inferential Statistics and Hypothesis Testing

**Hypothesis Testing Framework:**
1. Define **H₀** (null hypothesis) and **H₁** (alternative hypothesis)
2. Choose significance level **α** (usually 0.05)
3. Run the test and get the **p-value**
4. If **p-value < α** → reject H₀ (result is statistically significant)

```python
from scipy import stats

# T-Test — compare means of two independent groups
# H₀: Group A and Group B have the same mean salary
# H₁: Their means differ

group_a = df[df['Department'] == 'IT']['Salary']
group_b = df[df['Department'] == 'HR']['Salary']

t_stat, p_value = stats.ttest_ind(group_a, group_b)
print(f"T-statistic: {t_stat:.4f}, P-value: {p_value:.4f}")

if p_value < 0.05:
    print("Reject H₀: Significant difference in salaries")
else:
    print("Fail to reject H₀: No significant difference")

# One-sample T-test — compare sample mean to a known value
t_stat, p_value = stats.ttest_1samp(df['Salary'], popmean=70000)

# Paired T-test — compare means of the same group at different times
t_stat, p_value = stats.ttest_rel(before_scores, after_scores)

# ANOVA — compare means of THREE OR MORE groups
# H₀: All group means are equal
group_it      = df[df['Department'] == 'IT']['Salary']
group_hr      = df[df['Department'] == 'HR']['Salary']
group_finance = df[df['Department'] == 'Finance']['Salary']

f_stat, p_value = stats.f_oneway(group_it, group_hr, group_finance)
print(f"F-statistic: {f_stat:.4f}, P-value: {p_value:.4f}")

# Chi-Square Test — test relationship between two categorical variables
# H₀: Variables are independent
from scipy.stats import chi2_contingency

contingency_table = pd.crosstab(df['Department'], df['Gender'])
chi2, p, dof, expected = chi2_contingency(contingency_table)
```

---

### Correlation Analysis

```python
# Pearson correlation — linear relationship (default)
corr = df['Years_Experience'].corr(df['Salary'])
print(f"Pearson r = {corr:.4f}")
# r close to 1 = strong positive, -1 = strong negative, 0 = no linear relationship

# Spearman correlation — monotonic relationship (rank-based, handles outliers better)
spearman_corr, p_value = stats.spearmanr(df['Experience'], df['Salary'])

# Full correlation matrix
print(df.select_dtypes(include='number').corr())

# Visualize with heatmap
import seaborn as sns
import matplotlib.pyplot as plt

sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.show()
```

---

## 6. Machine Learning with Scikit-Learn

### What is Scikit-Learn?

**Scikit-Learn** is Python's most popular machine learning library. It provides:
- Consistent API (`fit`, `predict`, `transform`) for all algorithms
- Tools for model evaluation and selection
- Preprocessing utilities
- Supports classification, regression, clustering, dimensionality reduction

---

### Supervised vs Unsupervised Learning

| Type | Has Labels? | Goal | Examples |
|------|------------|------|---------|
| Supervised | Yes | Predict a target variable | Linear Regression, Decision Trees, SVM |
| Unsupervised | No | Discover patterns/structure | K-Means, PCA, DBSCAN |
| Semi-supervised | Partial | Combine both | Label propagation |

---

### The ML Workflow

```python
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd

# 1. Load data
df = pd.read_csv('housing.csv')
X = df.drop('Price', axis=1)      # Features
y = df['Price']                    # Target

# 2. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# test_size=0.2 → 80% train, 20% test

# 3. Scale features (important for distance-based algorithms)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit on train, transform train
X_test  = scaler.transform(X_test)        # only transform test (no fit!)

# 4. Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Make predictions
y_pred = model.predict(X_test)

# 6. Evaluate
mse  = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2   = r2_score(y_test, y_pred)
print(f"RMSE: {rmse:.2f}, R²: {r2:.4f}")
```

---

### Supervised Learning Algorithms

**Linear Regression** — Predicts a continuous value

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# Simple linear regression
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Coefficients: {model.coef_}")
print(f"Intercept: {model.intercept_}")

# Ridge (L2 regularization) — penalizes large coefficients
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)

# Lasso (L1 regularization) — can zero out coefficients (feature selection)
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
```

**Logistic Regression** — Predicts a binary/categorical outcome

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred       = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # probability of class 1
```

**Decision Trees**

```python
from sklearn.tree import DecisionTreeClassifier, export_text

model = DecisionTreeClassifier(max_depth=5, min_samples_split=10, random_state=42)
model.fit(X_train, y_train)

# Visualize feature importance
import pandas as pd
importances = pd.Series(model.feature_importances_, index=feature_names)
importances.sort_values().plot(kind='barh')
```

**Random Forest** — Ensemble of many decision trees

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_clf.fit(X_train, y_train)

# Regression
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train, y_train)
```

---

### Model Evaluation Metrics

**For Classification:**

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

y_pred = model.predict(X_test)

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.4f}")
print(f"AUC-ROC:   {roc_auc_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

| Metric | Formula | When to use |
|--------|---------|------------|
| Accuracy | Correct / Total | Balanced classes |
| Precision | TP / (TP + FP) | When false positives are costly (e.g., spam) |
| Recall | TP / (TP + FN) | When false negatives are costly (e.g., cancer detection) |
| F1 Score | 2 × P×R / (P+R) | Imbalanced classes |
| AUC-ROC | Area under ROC curve | Overall model quality |

**For Regression:**

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

mse  = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae  = mean_absolute_error(y_test, y_pred)
r2   = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")   # Same unit as target; lower is better
print(f"MAE:  {mae:.2f}")    # Less sensitive to outliers than RMSE
print(f"R²:   {r2:.4f}")     # 1.0 = perfect; 0 = as good as predicting mean
```

---

### Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

kf     = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')

print(f"CV R² scores: {scores}")
print(f"Mean R²: {scores.mean():.4f} ± {scores.std():.4f}")
```

---

### Unsupervised Learning Algorithms

**K-Means Clustering**

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Find optimal number of clusters using the Elbow Method
inertias = []
k_range  = range(1, 11)
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)

plt.plot(k_range, inertias, 'bo-')
plt.xlabel('Number of clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method')
plt.show()

# Fit with chosen k
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)
df['Cluster'] = kmeans.labels_
```

**Principal Component Analysis (PCA) — Dimensionality Reduction**

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance retained: {pca.explained_variance_ratio_.sum():.2%}")

# Visualize
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA — First 2 Components')
plt.show()
```

---

## 7. Time Series Analysis

### What is Time Series Analysis?

Time series analysis studies data collected at **successive points in time** to find patterns, trends, and seasonal cycles — and to make **forecasts**.

Common applications: stock prices, sales forecasting, weather, web traffic, energy consumption.

---

### Working with Time Series Data in Pandas

```python
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('sales.csv')

# Convert the date column to datetime
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Explore
print(df.index.min(), df.index.max())   # date range
print(df.resample('M').mean())          # monthly average
print(df.resample('Q').sum())           # quarterly sum
print(df.resample('Y').sum())           # annual sum

# Resampling — change frequency
daily_to_monthly = df['Sales'].resample('M').sum()   # aggregate daily → monthly
monthly_to_weekly = df['Sales'].resample('W').mean() # interpolate monthly → weekly

# Rolling averages — smooth out noise
df['MA_7']  = df['Sales'].rolling(window=7).mean()   # 7-day moving average
df['MA_30'] = df['Sales'].rolling(window=30).mean()  # 30-day moving average

df[['Sales', 'MA_7', 'MA_30']].plot(figsize=(12, 5))
plt.title('Sales with Moving Averages')
plt.show()
```

---

### Seasonality and Trend Decomposition

```python
from statsmodels.tsa.seasonal import seasonal_decompose

# Decompose into: Trend + Seasonality + Residual
result = seasonal_decompose(df['Sales'], model='additive', period=12)
# model='additive'      if seasonality doesn't change with trend level
# model='multiplicative' if seasonality scales with trend level

result.plot()
plt.tight_layout()
plt.show()

# Access components
trend    = result.trend
seasonal = result.seasonal
residual = result.resid
```

---

### Stationarity

Most time series models require **stationarity** (constant mean, variance, and autocorrelation over time).

```python
from statsmodels.tsa.stattools import adfuller

# Augmented Dickey-Fuller Test
# H₀: Series is NON-stationary (has unit root)
# H₁: Series IS stationary
result = adfuller(df['Sales'])
print(f"ADF Statistic: {result[0]:.4f}")
print(f"P-value: {result[1]:.4f}")

if result[1] < 0.05:
    print("Series is STATIONARY (reject H₀)")
else:
    print("Series is NON-STATIONARY — try differencing")
    df['Sales_diff'] = df['Sales'].diff()   # First differencing
```

---

### Forecasting with ARIMA

**ARIMA(p, d, q):**
- **p** = autoregressive order (how many past values to use)
- **d** = differencing order (to make series stationary)
- **q** = moving average order (past forecast errors)

```python
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
import numpy as np

# Split into train/test
train = df['Sales'][:-12]   # all but last 12 months
test  = df['Sales'][-12:]   # last 12 months

# Fit ARIMA model
model  = ARIMA(train, order=(1, 1, 1))
result = model.fit()

print(result.summary())

# Forecast
forecast = result.forecast(steps=12)
print(forecast)

# Evaluate
rmse = np.sqrt(mean_squared_error(test, forecast))
print(f"RMSE: {rmse:.2f}")

# Plot
plt.figure(figsize=(12, 5))
plt.plot(train, label='Train')
plt.plot(test, label='Actual')
plt.plot(forecast, label='Forecast', linestyle='--')
plt.legend()
plt.show()
```

---

### Exponential Smoothing (ETS)

```python
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Triple exponential smoothing (Holt-Winters) — handles trend and seasonality
model = ExponentialSmoothing(
    train,
    trend='add',             # additive trend
    seasonal='add',          # additive seasonality
    seasonal_periods=12      # 12-month seasonality
)
result  = model.fit()
forecast = result.forecast(steps=12)

plt.plot(train, label='Train')
plt.plot(test, label='Actual')
plt.plot(forecast, label='ETS Forecast', linestyle='--')
plt.legend()
plt.show()
```

---

## 8. Web Scraping with BeautifulSoup & Requests

### What is web scraping?

Web scraping is the automated extraction of data from websites. It's used when data isn't available via an API or for download.

**Legal Note:** Always check a site's `robots.txt` and terms of service before scraping.

---

### Making HTTP Requests

```python
import requests

# GET request
response = requests.get('https://example.com')

print(response.status_code)   # 200 = OK, 404 = Not Found, 403 = Forbidden
print(response.text)          # Raw HTML
print(response.headers)       # Response headers

# With headers (to avoid being blocked as a bot)
headers  = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
response = requests.get('https://example.com', headers=headers, timeout=10)

# POST request (e.g., submitting a form)
payload  = {'username': 'user', 'password': 'pass'}
response = requests.post('https://example.com/login', data=payload)
```

---

### Parsing HTML with BeautifulSoup

```python
from bs4 import BeautifulSoup

response = requests.get('https://quotes.toscrape.com')
soup = BeautifulSoup(response.text, 'html.parser')

# Find elements
title      = soup.title.text                         # Page title
h1_tag     = soup.find('h1')                         # First <h1>
all_h2     = soup.find_all('h2')                     # All <h2> tags
paragraphs = soup.find_all('p')

# Find by class or id
quote_div  = soup.find('div', class_='quote')
nav_id     = soup.find(id='navigation')

# Extract text
print(soup.find('span', class_='text').text)

# Extract attributes
img_url  = soup.find('img')['src']
link_url = soup.find('a')['href']

# CSS selectors (very powerful)
quotes     = soup.select('.quote .text')
first_link = soup.select_one('a.nav-link')
```

---

### Full Scraping Example

```python
import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def scrape_quotes(num_pages=3):
    all_quotes = []
    
    for page in range(1, num_pages + 1):
        url      = f'https://quotes.toscrape.com/page/{page}/'
        response = requests.get(url, timeout=10)
        
        if response.status_code != 200:
            print(f"Failed to fetch page {page}")
            break
        
        soup   = BeautifulSoup(response.text, 'html.parser')
        quotes = soup.find_all('div', class_='quote')
        
        for quote in quotes:
            text   = quote.find('span', class_='text').text.strip()
            author = quote.find('small', class_='author').text.strip()
            tags   = [tag.text for tag in quote.find_all('a', class_='tag')]
            
            all_quotes.append({
                'text':   text,
                'author': author,
                'tags':   ', '.join(tags)
            })
        
        time.sleep(1)   # Be polite — wait between requests
    
    return pd.DataFrame(all_quotes)

df = scrape_quotes(num_pages=3)
df.to_csv('quotes.csv', index=False)
print(df.head())
```

---

### Handling Dynamic Content with Selenium

When a site loads content via JavaScript, `requests` won't see it. Use **Selenium**.

```python
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time

driver = webdriver.Chrome()
driver.get('https://example.com')

# Wait for element to appear (up to 10 seconds)
wait = WebDriverWait(driver, 10)
element = wait.until(EC.presence_of_element_located((By.CLASS_NAME, 'content')))

# Scroll to load more content
driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
time.sleep(2)

# Extract
items = driver.find_elements(By.CLASS_NAME, 'item')
for item in items:
    print(item.text)

driver.quit()
```

---

## 9. SQL for Data Analysis (Python Integration)

### Connecting to a Database from Python

```python
import sqlite3
import pandas as pd

# SQLite (no server needed — great for local files)
conn   = sqlite3.connect('database.db')
cursor = conn.cursor()

# Execute a query
cursor.execute("SELECT * FROM employees WHERE salary > 50000")
rows = cursor.fetchall()

# Better: read directly into a DataFrame
df = pd.read_sql_query("SELECT * FROM employees WHERE salary > 50000", conn)

conn.close()
```

```python
# PostgreSQL with psycopg2
import psycopg2

conn = psycopg2.connect(
    host='localhost', database='mydb',
    user='admin', password='secret'
)
df = pd.read_sql_query("SELECT * FROM sales", conn)
conn.close()
```

```python
# MySQL with SQLAlchemy (recommended for Pandas)
from sqlalchemy import create_engine

engine = create_engine('mysql+pymysql://user:password@localhost/mydb')
df = pd.read_sql('SELECT * FROM employees', con=engine)

# Write DataFrame back to SQL
df.to_sql('new_table', con=engine, if_exists='replace', index=False)
```

---

### Key SQL Commands for Data Analysts

```python
# Build queries dynamically
dept_id = 3
query = f"""
    SELECT e.name, d.department_name, e.salary
    FROM employees e
    INNER JOIN departments d ON e.department_id = d.id
    WHERE e.department_id = {dept_id}
    ORDER BY e.salary DESC
    LIMIT 10
"""
df = pd.read_sql_query(query, conn)
```

---

## 10. Advanced Data Visualization

### Interactive Visualizations with Plotly

```python
import plotly.express as px
import plotly.graph_objects as go

df = px.data.gapminder()

# Interactive scatter plot
fig = px.scatter(
    df[df['year'] == 2007],
    x='gdpPercap', y='lifeExp',
    color='continent',
    size='pop',
    hover_name='country',
    log_x=True,
    title='GDP vs Life Expectancy (2007)'
)
fig.show()

# Interactive line chart
fig = px.line(
    df[df['country'].isin(['USA', 'China', 'India'])],
    x='year', y='gdpPercap',
    color='country',
    title='GDP per Capita Over Time'
)
fig.show()

# Interactive bar chart
fig = px.bar(
    df[df['year'] == 2007].groupby('continent')['pop'].sum().reset_index(),
    x='continent', y='pop',
    title='Population by Continent'
)
fig.show()

# Choropleth map
fig = px.choropleth(
    df[df['year'] == 2007],
    locations='iso_alpha',
    color='lifeExp',
    hover_name='country',
    color_continuous_scale='RdYlGn',
    title='Life Expectancy World Map'
)
fig.show()
```

---

### Dash for Interactive Web Dashboards

```python
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
import pandas as pd

df = pd.read_csv('sales.csv')
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Sales Dashboard', style={'textAlign': 'center'}),
    dcc.Dropdown(
        id='category-dropdown',
        options=[{'label': c, 'value': c} for c in df['Category'].unique()],
        value=df['Category'].unique()[0],
        clearable=False
    ),
    dcc.Graph(id='sales-chart')
])

@app.callback(
    Output('sales-chart', 'figure'),
    Input('category-dropdown', 'value')
)
def update_chart(selected_category):
    filtered_df = df[df['Category'] == selected_category]
    fig = px.line(filtered_df, x='Date', y='Sales', title=f'Sales — {selected_category}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True)
```

---

### Geospatial Visualization with Folium

```python
import folium

# Create a map centered on a location
m = folium.Map(location=[40.7128, -74.0060], zoom_start=12)  # New York

# Add a marker
folium.Marker(
    location=[40.7128, -74.0060],
    popup='New York City',
    tooltip='Click me',
    icon=folium.Icon(color='blue', icon='info-sign')
).add_to(m)

# Add a circle
folium.Circle(
    location=[40.7128, -74.0060],
    radius=1000,           # meters
    color='crimson',
    fill=True
).add_to(m)

m.save('map.html')    # Open in browser
```

---

## 11. Natural Language Processing (NLP)

### What is NLP?

**Natural Language Processing** is the field of AI concerned with enabling computers to understand, interpret, and generate human language. Applications include chatbots, sentiment analysis, translation, summarization, and search engines.

---

### Text Preprocessing

```python
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
import string

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

text = "Natural Language Processing is fascinating! It enables computers to understand human language."

# Tokenization — split into words or sentences
words     = word_tokenize(text)       # ['Natural', 'Language', 'Processing', ...]
sentences = sent_tokenize(text)       # ['Natural Language Processing...', 'It enables...']

# Lowercasing
words_lower = [w.lower() for w in words]

# Remove punctuation
words_clean = [w for w in words_lower if w not in string.punctuation]

# Remove stopwords — common words with little meaning (the, is, in, ...)
stop_words = set(stopwords.words('english'))
filtered   = [w for w in words_clean if w not in stop_words]
print(filtered)   # ['natural', 'language', 'processing', 'fascinating', ...]

# Stemming — reduce to root form (fast but crude)
stemmer = PorterStemmer()
stemmed = [stemmer.stem(w) for w in filtered]
# 'processing' → 'process', 'enables' → 'enabl'

# Lemmatization — reduce to dictionary base form (slower but accurate)
lemmatizer = WordNetLemmatizer()
lemmatized = [lemmatizer.lemmatize(w) for w in filtered]
# 'processing' → 'processing', 'enables' → 'enable'
```

---

### Text Analysis

```python
from nltk.probability import FreqDist

# Frequency distribution
freq_dist = FreqDist(filtered)
freq_dist.most_common(10)   # top 10 most common words
freq_dist.plot(20)          # visualize

# Word Cloud
from wordcloud import WordCloud
import matplotlib.pyplot as plt

text_all = ' '.join(filtered)
wc = WordCloud(width=800, height=400, background_color='white').generate(text_all)
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.show()

# TF-IDF — term importance across documents
from sklearn.feature_extraction.text import TfidfVectorizer

documents = [
    "Python is great for data science",
    "Data science uses Python and statistics",
    "Statistics is important for data analysis"
]

vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix = vectorizer.fit_transform(documents)
df_tfidf = pd.DataFrame(
    tfidf_matrix.toarray(),
    columns=vectorizer.get_feature_names_out()
)
```

---

### Sentiment Analysis

```python
# VADER — specifically designed for social media text
from nltk.sentiment import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')

sia = SentimentIntensityAnalyzer()

texts = [
    "I love this product! It's amazing!",
    "This is terrible. Worst purchase ever.",
    "The product is okay. Nothing special."
]

for text in texts:
    scores = sia.polarity_scores(text)
    print(f"Text: {text[:40]}...")
    print(f"  Compound: {scores['compound']:.3f}  Pos: {scores['pos']:.3f}  Neg: {scores['neg']:.3f}")
    # Compound > 0.05 = positive, < -0.05 = negative, else neutral
```

---

### Named Entity Recognition (NER)

```python
import spacy

nlp = spacy.load('en_core_web_sm')

text = "Apple Inc. was founded by Steve Jobs in Cupertino, California in 1976."
doc  = nlp(text)

for ent in doc.ents:
    print(f"{ent.text:20} → {ent.label_} ({spacy.explain(ent.label_)})")

# Apple Inc.           → ORG (Companies, agencies, institutions)
# Steve Jobs           → PERSON (People, including fictional)
# Cupertino            → GPE (Geopolitical entity — cities, states, countries)
# California           → GPE
# 1976                 → DATE
```

---

### Topic Modeling with LDA

```python
from gensim import corpora, models
from gensim.utils import simple_preprocess

documents = [
    "Machine learning models predict outcomes from data",
    "Deep learning uses neural networks for complex tasks",
    "Natural language processing handles text and speech",
    "Computer vision analyzes images and video",
    "Python is popular for machine learning and data science"
]

# Preprocess
processed = [simple_preprocess(doc) for doc in documents]

# Build dictionary and corpus
dictionary = corpora.Dictionary(processed)
corpus     = [dictionary.doc2bow(doc) for doc in processed]

# Train LDA model
lda_model = models.LdaModel(
    corpus, num_topics=2, id2word=dictionary, passes=15, random_state=42
)

# Print topics
for idx, topic in lda_model.print_topics(-1):
    print(f"Topic {idx}: {topic}")
```

---

## 12. Deep Learning with TensorFlow

### What is Deep Learning?

**Deep Learning** is a subset of ML using **neural networks with many layers** to learn complex representations of data. It excels at image recognition, NLP, speech recognition, and generative AI.

**TensorFlow** (by Google) and **PyTorch** (by Meta) are the two most popular frameworks.

---

### Building a Neural Network with Keras

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load sample data
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train / 255.0   # Normalize pixel values to [0, 1]
X_test  = X_test  / 255.0

# Build a Sequential model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),    # Flatten 28x28 image → 784 values
    keras.layers.Dense(128, activation='relu'),     # Hidden layer 1
    keras.layers.Dropout(0.2),                      # Regularization — prevent overfitting
    keras.layers.Dense(64, activation='relu'),      # Hidden layer 2
    keras.layers.Dense(10, activation='softmax')    # Output layer — 10 classes
])

# Compile
model.compile(
    optimizer='adam',                    # Adaptive learning rate optimizer
    loss='sparse_categorical_crossentropy',  # for integer labels
    metrics=['accuracy']
)

model.summary()   # Display architecture

# Train
history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.1,   # 10% of training data for validation
    verbose=1
)

# Evaluate
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

# Predict
predictions = model.predict(X_test[:5])
predicted_classes = np.argmax(predictions, axis=1)
```

---

### Convolutional Neural Networks (CNNs) — for Images

```python
model = keras.Sequential([
    # Convolutional layers — detect features (edges, shapes, textures)
    keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    keras.layers.MaxPooling2D((2, 2)),    # Reduce spatial dimensions

    keras.layers.Conv2D(64, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D((2, 2)),

    # Flatten and classify
    keras.layers.Flatten(),
    keras.layers.Dense(64, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
```

---

### Recurrent Neural Networks (RNNs) & LSTMs — for Sequences

```python
# LSTMs handle long-term dependencies in sequences (text, time series)
model = keras.Sequential([
    keras.layers.Embedding(input_dim=vocab_size, output_dim=64),  # word → vector
    keras.layers.LSTM(128, return_sequences=True),                  # sequence processing
    keras.layers.LSTM(64),
    keras.layers.Dense(1, activation='sigmoid')                     # binary output
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```

---

### Activation Functions Comparison

| Function | Formula | Use Case |
|---------|---------|---------|
| ReLU | max(0, x) | Hidden layers — default choice |
| Sigmoid | 1/(1+e⁻ˣ) | Binary classification output |
| Softmax | eˣⁱ/Σeˣ | Multi-class output (probabilities sum to 1) |
| Tanh | (eˣ-e⁻ˣ)/(eˣ+e⁻ˣ) | RNNs, normalized outputs |
| Leaky ReLU | max(αx, x) | Avoids "dying ReLU" problem |

---

## 13. Transfer Learning with Pre-trained Models

### What is Transfer Learning?

**Transfer Learning** reuses a model trained on a large dataset (source task) for a new, related task (target task). Instead of training from scratch, you:
1. Take a pre-trained model (e.g., VGG16 trained on ImageNet with 14M images)
2. Freeze its layers (keep learned features)
3. Add new layers on top for your specific task
4. Train only the new layers (or fine-tune all layers)

**Why use it?**
- Requires far less data and compute
- Achieves high accuracy quickly
- Leverages knowledge from huge datasets

---

### Image Classification with Transfer Learning

```python
import tensorflow as tf
from tensorflow import keras

# Load MobileNetV2 pre-trained on ImageNet
base_model = keras.applications.MobileNetV2(
    weights='imagenet',
    include_top=False,       # Exclude the final classification layer
    input_shape=(224, 224, 3)
)

# Freeze pre-trained layers — don't update their weights
base_model.trainable = False

# Add custom classification head
model = keras.Sequential([
    base_model,
    keras.layers.GlobalAveragePooling2D(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(5, activation='softmax')   # 5 custom classes
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Phase 1: Train only the new head
model.fit(train_data, validation_data=val_data, epochs=10)

# Phase 2: Fine-tune — unfreeze top layers of base model
base_model.trainable = True
for layer in base_model.layers[:-20]:   # Freeze all but last 20 layers
    layer.trainable = False

# Use a lower learning rate for fine-tuning
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_data, validation_data=val_data, epochs=5)
```

---

### NLP Transfer Learning — Fine-tuning BERT

```python
from transformers import BertTokenizer, TFBertForSequenceClassification
import tensorflow as tf

# Load pre-trained BERT
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model     = TFBertForSequenceClassification.from_pretrained(
    'bert-base-uncased', num_labels=2  # binary sentiment
)

# Tokenize text
texts = ["I love this product!", "This is terrible."]
encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors='tf')

# Predict (zero-shot inference)
outputs = model(encodings)
predictions = tf.nn.softmax(outputs.logits, axis=-1)
print(predictions)   # [[p_negative, p_positive], ...]
```

---

### Popular Pre-trained Models

| Model | Domain | Dataset | Use Cases |
|-------|--------|---------|---------|
| VGG16/19 | Images | ImageNet | Image classification |
| ResNet50 | Images | ImageNet | Image classification, detection |
| MobileNetV2 | Images | ImageNet | Mobile/edge deployment |
| InceptionV3 | Images | ImageNet | High accuracy image tasks |
| BERT | Text | BooksCorpus, Wikipedia | Classification, QA, NER |
| GPT-3/4 | Text | Web data | Generation, completion |
| RoBERTa | Text | Larger corpus | Improved BERT accuracy |
| Whisper | Audio | Multilingual audio | Speech-to-text |

---

## 14. Big Data Processing with Apache Spark

### What is Apache Spark?

**Apache Spark** is an open-source distributed computing system designed to process **massive datasets** across a cluster of machines at high speed. It can process data up to 100x faster than Hadoop MapReduce by using in-memory computation.

**PySpark** is the Python API for Spark.

**Note:** Spark is typically used when data is too large for a single machine (GBs to PBs). For most data analyst roles, Pandas is sufficient.

---

### Core Concepts

```python
from pyspark import SparkContext
from pyspark.sql import SparkSession

# Create a Spark session (entry point for PySpark)
spark = SparkSession.builder \
    .appName("DataAnalysis") \
    .master("local[*]") \
    .getOrCreate()

sc = spark.sparkContext   # Lower-level API
```

---

### RDDs (Resilient Distributed Datasets)

RDDs are the fundamental data structure of Spark — an immutable, distributed collection of objects.

```python
# Create an RDD
rdd = sc.parallelize([1, 2, 3, 4, 5])

# Transformations — return a new RDD (lazy — not executed until an action is called)
squared_rdd = rdd.map(lambda x: x ** 2)                # [1, 4, 9, 16, 25]
even_rdd    = rdd.filter(lambda x: x % 2 == 0)         # [2, 4]
flat_rdd    = rdd.flatMap(lambda x: [x, x * 2])        # [1,2, 2,4, 3,6, ...]

# Actions — trigger actual computation
print(squared_rdd.collect())    # [1, 4, 9, 16, 25] — bring all data to driver
print(rdd.count())              # 5
print(rdd.sum())                # 15
print(rdd.reduce(lambda a, b: a + b))  # 15

# Word count classic example
text_rdd   = sc.textFile("data.txt")
words      = text_rdd.flatMap(lambda line: line.split())
word_pairs = words.map(lambda word: (word, 1))
word_count = word_pairs.reduceByKey(lambda a, b: a + b)
print(word_count.collect())
```

---

### PySpark DataFrames (Recommended over RDDs)

PySpark DataFrames are optimized and have a SQL-like API similar to Pandas.

```python
# Read data
df = spark.read.csv("employees.csv", header=True, inferSchema=True)
df = spark.read.json("data.json")
df = spark.read.parquet("data.parquet")   # Parquet is preferred for big data

# Explore
df.printSchema()        # Column names and types
df.show(5)              # First 5 rows
df.count()              # Row count
df.describe().show()    # Summary statistics

# Select and filter
df.select('name', 'salary').show()
df.filter(df['salary'] > 60000).show()
df.filter("department = 'IT'").show()

# GroupBy and aggregation
from pyspark.sql import functions as F

df.groupBy('department').agg(
    F.avg('salary').alias('avg_salary'),
    F.count('*').alias('headcount'),
    F.max('salary').alias('max_salary')
).show()

# Add a computed column
df = df.withColumn('salary_k', df['salary'] / 1000)
df = df.withColumn(
    'level',
    F.when(df['salary'] > 80000, 'Senior')
     .when(df['salary'] > 60000, 'Mid')
     .otherwise('Junior')
)

# Sort
df.orderBy('salary', ascending=False).show()

# Join
df_emp  = spark.read.csv("employees.csv", header=True, inferSchema=True)
df_dept = spark.read.csv("departments.csv", header=True, inferSchema=True)
joined  = df_emp.join(df_dept, on='department_id', how='inner')
```

---

### Spark SQL

```python
# Register DataFrame as a temporary SQL table
df.createOrReplaceTempView("employees")

# Run SQL queries on it
result = spark.sql("""
    SELECT department, AVG(salary) AS avg_salary, COUNT(*) AS headcount
    FROM employees
    WHERE salary > 50000
    GROUP BY department
    HAVING COUNT(*) > 5
    ORDER BY avg_salary DESC
""")
result.show()
```

---

### Spark MLlib — Distributed Machine Learning

```python
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator

# Assemble features into a single vector column
assembler = VectorAssembler(
    inputCols=['experience', 'education_score'],
    outputCol='features'
)

# Define model
lr = LinearRegression(featuresCol='features', labelCol='salary')

# Build pipeline
pipeline = Pipeline(stages=[assembler, lr])

# Split data
train, test = df.randomSplit([0.8, 0.2], seed=42)

# Train
model = pipeline.fit(train)

# Predict
predictions = model.transform(test)

# Evaluate
evaluator = RegressionEvaluator(labelCol='salary', predictionCol='prediction', metricName='rmse')
rmse = evaluator.evaluate(predictions)
print(f"RMSE: {rmse:.2f}")
```

---

### Spark vs Pandas Comparison

| Feature | Pandas | PySpark |
|--------|--------|---------|
| Data size | Up to ~GB (single machine) | TB to PB (distributed cluster) |
| Execution | Eager (immediate) | Lazy (optimized execution plan) |
| Speed | Fast for small data | Fast for large data |
| API familiarity | Intuitive | Similar but some differences |
| Machine learning | Scikit-Learn | MLlib |
| Setup complexity | Simple (pip install) | Requires Spark cluster or local setup |
| Use case | Data analysis, prototyping | Production big data pipelines |

**When to use Spark:** When your data doesn't fit in memory, or when you need to process data at scale across a distributed cluster.

---

## Quick Reference Cheatsheet

```python
# ══════════════════════════════════════
# NUMPY ESSENTIALS
# ══════════════════════════════════════
import numpy as np
arr = np.array([1, 2, 3])
arr.reshape(1, 3)           # Change shape
arr[arr > 2]                # Boolean indexing
np.mean(arr), np.std(arr)  # Statistics
np.dot(A, B)               # Matrix multiplication
np.where(arr > 2, 1, 0)    # Conditional replace

# ══════════════════════════════════════
# PANDAS ESSENTIALS
# ══════════════════════════════════════
import pandas as pd
df = pd.read_csv('file.csv')
df.info(), df.describe(), df.head()
df['col'].value_counts()
df[df['col'] > 5]                    # Filter
df.groupby('col').agg({'val': 'mean'}) # Group
df.merge(df2, on='key', how='left')  # Join
df.dropna(), df.fillna(0)            # Handle NaN
df.drop_duplicates()                 # Remove dupes

# ══════════════════════════════════════
# MATPLOTLIB/SEABORN ESSENTIALS
# ══════════════════════════════════════
import matplotlib.pyplot as plt
import seaborn as sns
plt.plot(x, y); plt.show()           # Line plot
plt.bar(x, y)                        # Bar chart
sns.heatmap(df.corr(), annot=True)  # Heatmap
sns.pairplot(df)                     # Pair plot
sns.boxplot(x='cat', y='num', data=df) # Box plot

# ══════════════════════════════════════
# SKLEARN ESSENTIALS
# ══════════════════════════════════════
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ══════════════════════════════════════
# DATA CLEANING CHECKLIST
# ══════════════════════════════════════
df.isnull().sum()                    # Missing values
df.fillna(df.mean())                 # Fill with mean
df.drop_duplicates()                 # Remove duplicates
df.dtypes / df.astype(int)          # Type checking
df['col'].str.strip().str.lower()   # String cleaning
df['date'] = pd.to_datetime(df['date']) # Parse dates
IQR = Q3 - Q1; clip outliers        # Outlier handling
MinMaxScaler() / StandardScaler()   # Scale features
pd.get_dummies() / LabelEncoder()   # Encode categories
```

---

*Master these libraries through hands-on practice on real datasets. Kaggle, UCI ML Repository, and Google Colab are excellent free resources for practicing Python data analysis.*
