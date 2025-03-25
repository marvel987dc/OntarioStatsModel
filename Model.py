# -*- coding: utf-8 -*-
"""
Created on Wed Mar 19 20:20:45 2025

@author: Juan David
"""

#these are the imports we will use for the model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from parso.python.tokenize import group
import seaborn as sns


#data exploring

# Load the dataset
file_path = r"C:\Users\barre\Documents\Semester 4 (Current)\Supervised Learning (SEC. 001)\KSI_Group_Group#_section_section#COMP247Project\Data\KilledAndInjured.csv"
Group_data = pd.read_csv(file_path)
# --------------------------------------------------------------
print(f"Original dataset now has {Group_data.shape[0]} rows.")
# Find and count duplicate rows (excluding the first occurrence)
duplicate_rows = Group_data[Group_data.duplicated(keep=False)]  # Keeps all duplicates

# Count how many times each duplicate appears (Fixed)
duplicate_counts = duplicate_rows.groupby(duplicate_rows.columns.tolist()).size()

# Print the result
num_duplicates = duplicate_rows.shape[0]

if num_duplicates > 0:
    print(f"Found {num_duplicates} duplicate rows.\n")
    print("Duplicate Rows and Their Counts:")
    print(duplicate_counts)  # Shows duplicate rows and their occurrences
else:
    print("No duplicate rows found. The dataset does not contain exact copies.")

# Remove duplicate rows from the dataset
Group_data = Group_data.drop_duplicates()

# Confirm removal
print(f"Cleaned dataset now has {Group_data.shape[0]} rows.")
print("Duplicate rows successfully removed.")
# --------------------------------------------------------------
# Display the first five rows of the dataset
print("\nDataset Information:")
print(Group_data.info())

# Display the first five rows of the dataset
print("\nFirst Five Rows:")
print(Group_data.head())

# Display the last five rows of the dataset
print("\nColumn names and data types: ")
print(Group_data.dtypes)

# Display the last five rows of the dataset
print("\nSummary Statistics: ")
print(Group_data.describe())

# Display the last five rows of the dataset
print("\nUnique Values per column: ")
# Display unique values for each column if the number of unique values is less than or equal to 15
for col in Group_data.columns:
    unique_values = Group_data[col].nunique()
    if unique_values <= 15:
        print(col + " : " + str(unique_values) + " - " + str(Group_data[col].unique()))

# Display the last five rows of the dataset
print("\nRanges of Numeric Columns: ")
# Display the range of numeric columns min and max values
for col in Group_data.select_dtypes(include=np.number).columns:
    print(col + " : " + str(Group_data[col].min()) + " - " + str(Group_data[col].max()))

# select the numeric_columns from the dataset
# Here we are filtering all the numeric columns from the dataset before the calculations
numeric_cols = Group_data.select_dtypes(include=['int64', 'float64'])

# displays the mean
print("\nMean (Average) of Numeric Columns:")
print(numeric_cols.mean())

# Calculate the median for numeric columns
print("\nMedian of Numeric Columns:")
print(numeric_cols.median())

# Calculate the standard deviation for numeric columns
print("\nStandard Deviation of Numeric Columns:")
print(numeric_cols.std())

# Calculate correlations between numeric columns
# print("\n🔗 Correlation Between Numeric Columns:")
# Group_data.corr()

# Calculate correlations between numeric columns
correlations = numeric_cols.corr()

print("\nCorrelation Between Numeric Columns:")
print(correlations)

# Calculate correlations between numeric columns using the Spearman method
spearman_corr = numeric_cols.corr(method='spearman')
print("\nSpearman Correlation Between Numeric Columns:")
print(spearman_corr)

# differences between the normal correlation:
# and spearman correlation Measures the linear relationship between two variables.
# Spearman  Measures the monotonic relationship between two variables (whether the relationship is consistently increasing or decreasing,
# but not necessarily linear).

# display the missing values in the dataset
print("\nNumber of missing values: ")
missing_values = Group_data.isnull().sum()

# calculate the percentage of missing values, this is the formula
missing_percentage = (missing_values / len(Group_data)) * 100

# displaying the results and storing them in a dataframe for better visualization
missing_data_summary = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})

# display the missing values if there are any in the dataset and sort them in descending order to see the most missing values
print("\n Missing Data Summary:")
print(missing_data_summary[missing_data_summary['Missing Values'] > 0].sort_values('Percentage', ascending=False))

# -----------------------------------------------------------------------------
# Graphs and visualizations

# Set seaborn style
sns.set(style="whitegrid")

# Identify the target variable (assuming a column indicates fatality)
fatality_columns = [col for col in Group_data.columns if 'fatal' in col.lower() or 'death' in col.lower()]
target_col = fatality_columns[0] if fatality_columns else None

# Plot the distribution of the target variable (if found)
# Convert FATAL_NO into a binary target variable (Fatal: 1, Non-Fatal: 0)
Group_data["Fatal_Collision"] = Group_data["FATAL_NO"].fillna(0).astype(float)  # Convert NaN to 0
Group_data["Fatal_Collision"] = (Group_data["Fatal_Collision"] > 0).astype(int)  # Convert to binary (1 if fatal, else 0)

# Plot the cleaned distribution
plt.figure(figsize=(6, 4))
sns.countplot(x=Group_data["Fatal_Collision"], palette="coolwarm")
plt.title("Distribution of Fatal vs. Non-Fatal Collisions")
plt.xlabel("Collision Outcome")
plt.ylabel("Count")
plt.xticks([0, 1], ["Non-Fatal", "Fatal"])  # Ensure readable labels
plt.show()

# Correlation heatmap
# Select only numeric columns (exclude text, dates, etc.)
numeric_cols = Group_data.select_dtypes(include=['int64', 'float64'])
# Generate Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap")
plt.show()

# Missing values heatmap
plt.figure(figsize=(10, 6))
sns.heatmap(Group_data.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Missing Values Heatmap")
plt.show()

# Selecting numeric columns for histograms and boxplots
numeric_cols = Group_data.select_dtypes(include=['int64', 'float64']).columns

# Histograms for numerical features
Group_data[numeric_cols].hist(figsize=(15, 10), bins=20, color="steelblue", edgecolor="black")
plt.suptitle("Distribution of Numerical Features", fontsize=16)
plt.show()

# Select relevant numeric columns (excluding IDs and coordinates)
exclude_cols = ["OBJECTID", "INDEX", "ACCNUM", "LATITUDE", "LONGITUDE", "x", "y"]
filtered_numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

# Limit to the top 8 numerical features to avoid clutter
top_features = filtered_numeric_cols[:8]

# Create separate vertical boxplots
plt.figure(figsize=(10, 6))
sns.boxplot(data=Group_data[top_features], palette="coolwarm")
plt.xticks(rotation=45)  # Rotate labels for better readability
plt.title("Boxplots of Key Numeric Features")
plt.show()

# Cleaning the numeric dataset for pairplot
clean_numeric_cols = [col for col in numeric_cols if
                      Group_data[col].notna().all() and np.isfinite(Group_data[col]).all()]
subset_cols = clean_numeric_cols[:5]  # Taking the first five cleaned numerical features for pairplot

# Pairplot for a subset of numeric columns (if enough valid numeric columns exist)
if len(subset_cols) > 1:
    sns.pairplot(Group_data[subset_cols], diag_kind="kde", corner=True)
    plt.suptitle("Pairplot of Selected Numerical Features", fontsize=16)
    plt.show()
else:
    print("Not enough valid numeric columns available for pairplot.")

# -----------------------------------------------------------------------------
# 2. DATA MODELLING
# -----------------------------------------------------------------------------

# 2.1. Data transformations – Handling missing data, categorical data, normalization, standardization
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Identify categorical and numerical columns
categorical_cols = Group_data.select_dtypes(include=['object']).columns
numerical_cols = Group_data.select_dtypes(include=['int64', 'float64']).columns

# Handling missing numerical data (imputation)
num_imputer = SimpleImputer(strategy="median")

# Handling categorical data (encoding)
cat_imputer = SimpleImputer(strategy="most_frequent")  # Fill missing categorical values
one_hot_encoder = OneHotEncoder(handle_unknown="ignore")

# Normalization (MinMax Scaling) & Standardization (Standard Scaler)
scaler = StandardScaler()  # Change to MinMaxScaler() if needed

# Column Transformer: Applies transformations to different column types
preprocessor = ColumnTransformer([
    ("num", Pipeline([("imputer", num_imputer), ("scaler", scaler)]), numerical_cols),
    ("cat", Pipeline([("imputer", cat_imputer), ("encoder", one_hot_encoder)]), categorical_cols)
])

# -----------------------------------------------------------------------------
# 2.2. Feature selection – Choosing relevant columns
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif
import numpy as np
import pandas as pd

# Define input features and target variable
X = Group_data.drop(columns=["Fatal_Collision", "FATAL_NO"])  # Drop target column
y = Group_data["Fatal_Collision"]

# Identify non-numeric columns
non_numeric_cols = X.select_dtypes(exclude=['int64', 'float64']).columns

# Print columns that need conversion or removal
if len(non_numeric_cols) > 0:
    print("\nNon-numeric columns found (these must be removed or converted):")
    print(non_numeric_cols)

# Drop non-numeric columns (like timestamps) before feature selection
X_clean = X.drop(columns=non_numeric_cols)

# Check for NaN values before imputation
print("\nChecking for missing values before imputation:")
print(X_clean.isnull().sum()[X_clean.isnull().sum() > 0])

irrelevant_cols = ["OBJECTID", "INDEX", "ACCNUM"]
X_clean = X_clean.drop(columns=irrelevant_cols, errors="ignore")
# Handle missing values by filling NaN with the median (for numerical columns)
imputer = SimpleImputer(strategy="median")
X_imputed = pd.DataFrame(imputer.fit_transform(X_clean), columns=X_clean.columns)

# Check if any NaN values remain after imputation
if np.isnan(X_imputed).sum().sum() > 0:
    print("\nERROR: NaN values still present after imputation!")
    print(X_imputed.isnull().sum()[X_imputed.isnull().sum() > 0])
else:
    print("\nAll missing values successfully handled.")

# Ensure all values are finite (no NaN or Inf)
X_imputed = X_imputed.replace([np.inf, -np.inf], np.nan)  # Replace infinite values with NaN
X_imputed = X_imputed.dropna()  # Drop any rows that still contain NaN

# Feature selection using ANOVA F-test (Select Top 10 Features)
feature_selector = SelectKBest(score_func=f_classif, k=min(10, X_imputed.shape[1]))  # Ensure k ≤ total features
X_selected = feature_selector.fit_transform(X_imputed, y.loc[X_imputed.index])  # Ensure y is aligned

# Print selected feature scores
feature_scores = pd.DataFrame({"Feature": X_clean.columns, "Score": feature_selector.scores_})
print("\nFeature Selection Scores:")
print(feature_scores.sort_values(by="Score", ascending=False))

# -----------------------------------------------------------------------------
# 2.3. Train, Test data splitting – Using train_test_split
from sklearn.model_selection import train_test_split

# Splitting data into 80% training and 20% testing
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42, stratify=y)

print(f"Data Split: Training Set = {X_train.shape[0]} rows, Testing Set = {X_test.shape[0]} rows.")

# -----------------------------------------------------------------------------
# 2.4. Managing imbalanced classes – Oversampling / Undersampling
from imblearn.over_sampling import SMOTE
from collections import Counter

# Convert X_train back to a DataFrame if it's a NumPy array (before SMOTE)
if isinstance(X_train, np.ndarray):
    print("Warning: X_train is a NumPy array. Converting back to DataFrame...")
    X_train = pd.DataFrame(X_train, columns=X_clean.columns)  # Restore original feature names

# Save feature names before SMOTE
feature_names = X_train.columns

# Check class distribution before balancing
print("\nClass distribution before balancing:", Counter(y_train))

# Apply SMOTE for oversampling
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Convert X_train_balanced back to DataFrame with original column names
X_train_balanced = pd.DataFrame(X_train_balanced, columns=feature_names)

# Check class distribution after balancing
print("\nClass distribution after SMOTE balancing:", Counter(y_train_balanced))

# Apply feature selection correctly
selected_features = feature_selector.get_support(indices=True)

# Convert X_test to DataFrame (Ensure it supports `.iloc`)
if isinstance(X_test, np.ndarray):
    X_test = pd.DataFrame(X_test, columns=feature_names)

# Ensure that only selected features are used
X_train_balanced = X_train_balanced.iloc[:, selected_features]
X_test = X_test.iloc[:, selected_features]

# Verify columns before transforming
print("\nColumns in X_train_balanced after SMOTE and feature selection:", X_train_balanced.columns)
print("\nColumns in X_test after feature selection:", X_test.columns)

# -----------------------------------------------------------------------------
# 2.5. Using Pipelines to streamline preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

selected_feature_names = X_train_balanced.columns  # Get the correct column names

# Define preprocessing steps for numerical features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), selected_feature_names)  # Apply scaling only to selected features
    ]
)

# Create pipeline without feature selection (it was already applied)
pipeline = Pipeline([
    ("preprocessing", preprocessor)
])

# Fit-transform the training data
X_train_transformed = pipeline.fit_transform(X_train_balanced)

# Transform the test data
X_test_transformed = pipeline.transform(X_test)

print("\nPreprocessing pipeline applied successfully.")

print("\nShape of Transformed Training Data:", X_train_transformed.shape)
print("Shape of Transformed Test Data:", X_test_transformed.shape)
# Display first 5 rows (note: values are scaled, so they may look different)
print("\nPreview of Transformed Training Data:")
print(pd.DataFrame(X_train_transformed).head())
print("\nPreview of Transformed Test Data:")
print(pd.DataFrame(X_test_transformed).head())
