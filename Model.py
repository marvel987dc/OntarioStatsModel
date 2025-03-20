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

# Plot histograms for all numeric columns in the dataset to visualize the distribution of the data
Group_data.hist(figsize=(18, 14), bins=30, color='skyblue', edgecolor='black')
plt.suptitle("📊 Distribution of Numeric Variables", fontsize=18)
plt.show()

# Plot box plots for all numeric columns in the dataset to visualize the distribution of the data
plt.figure(figsize=(18, 14))
sns.boxplot(data=numeric_cols, palette='viridis')
plt.title("📦 Boxplot of Numeric Variables", fontsize=18)
plt.show()

# Calculate correlation
numeric_cols = Group_data.select_dtypes(include=['int64', 'float64'])
correlation_matrix = numeric_cols.corr()

# Plotting the heatmap
plt.figure(figsize=(18, 14))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title("🔥 Correlation Heatmap", fontsize=18)
plt.show()

# Sampling the data to avoid overload
sampled_data = Group_data.sample(500, random_state=53)

# Create a pairplot
sns.pairplot(sampled_data, hue='LIGHT', diag_kind='kde', palette='Set2')
plt.suptitle("🌟 Pairplot of Sampled Data", fontsize=18, y=1.02)
plt.show()

# Plotting latitude and longitude, plotting the accident locations
plt.figure(figsize=(12, 8))
sns.scatterplot(x='LONGITUDE', y='LATITUDE', data=Group_data, alpha=0.5, color='red')
plt.title("📍 Accident Locations by Latitude and Longitude")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# Vehicle involvement columns
vehicles = ['PEDESTRIAN', 'CYCLIST', 'AUTOMOBILE', 'MOTORCYCLE', 'TRUCK']

# Count the occurrences of each vehicle type
vehicle_counts = Group_data[vehicles].sum()

# Plot pie chart
plt.figure(figsize=(12, 8))
plt.pie(vehicle_counts, labels=vehicle_counts.index, autopct='%1.1f%%', startangle=140, colors=sns.color_palette("pastel"))
plt.title("🚗 Vehicle Involvement in Accidents")
plt.show()



