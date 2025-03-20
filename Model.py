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
print("\n Ranges of Numeric Columns: ")
# Display the range of numeric columns min and max values
for col in Group_data.select_dtypes(include=np.number).columns:
    print(col + " : " + str(Group_data[col].min()) + " - " + str(Group_data[col].max()))

# select the numeric_columns from the dataset
#Here we are filtering all the numeric columns from the dataset before the calculations
numeric_cols = Group_data.select_dtypes(include=['int64', 'float64'])

#displays the mean
print("\n📏 Mean (Average) of Numeric Columns:")
print(numeric_cols.mean())

# Calculate the median for numeric columns
print("\n📍 Median of Numeric Columns:")
print(numeric_cols.median())

# Calculate the standard deviation for numeric columns
print("\n📉 Standard Deviation of Numeric Columns:")
print(numeric_cols.std())

# Calculate correlations between numeric columns
# print("\n🔗 Correlation Between Numeric Columns:")
# Group_data.corr()

# Calculate correlations between numeric columns
correlations = numeric_cols.corr()

print("\n🔗 Correlation Between Numeric Columns:")
print(correlations)

# Calculate correlations between numeric columns using the Spearman method
spearman_corr = numeric_cols.corr(method='spearman')
print("\n🔗 Spearman Correlation Between Numeric Columns:")
print(spearman_corr)

#differences between the normal correlation:
# and spearman correlation Measures the linear relationship between two variables.
# Spearman  Measures the monotonic relationship between two variables (whether the relationship is consistently increasing or decreasing,
# but not necessarily linear).

#display the missing values in the dataset
print("\nNumber of missing values: ")
missing_values = Group_data.isnull().sum()

#calculate the percentage of missing values, this is the formula
missing_percentage = (missing_values / len(Group_data)) * 100

#displaying the results and storing them in a dataframe for better visualization
missing_data_summary = pd.DataFrame({
    'Missing Values': missing_values,
    'Percentage': missing_percentage
})

#display the missing values if there are any in the dataset and sort them in descending order to see the most missing values
print("\n🔍 Missing Data Summary:")
print(missing_data_summary[missing_data_summary['Missing Values'] > 0].sort_values('Percentage', ascending=False))

#plotting
sns.set(style="whitegrid")

# Identify the target variable (assuming a column indicates fatality)
fatality_columns = [col for col in Group_data.columns if 'fatal' in col.lower() or 'death' in col.lower()]
target_col = fatality_columns[0] if fatality_columns else None

# Plot the distribution of the target variable (if found)
# Convert FATAL_NO into a binary target variable (Fatal: 1, Non-Fatal: 0)
Group_data["Fatal_Collision"] = Group_data["FATAL_NO"].fillna(0).astype(float)  # Convert NaN to 0
Group_data["Fatal_Collision"] = (Group_data["Fatal_Collision"] > 0).astype(
    int)  # Convert to binary (1 if fatal, else 0)

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



