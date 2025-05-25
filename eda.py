# eda.py
import pandas as pd
import numpy as np

# Load dataset
df = pd.read_csv("House_Price_Prediction_Dataset.csv")  # replace with the correct filename

# Initial inspection
print("Initial shape:", df.shape)
print("Missing values:\n", df.isnull().sum())

# Drop rows with missing values (optional based on your dataset)
df.dropna(inplace=True)
print("Shape after dropping missing values:", df.shape)

# Convert categorical columns to string for clarity
df["Location"] = df["Location"].astype(str)
df["Condition"] = df["Condition"].astype(str)
df["Garage"] = df["Garage"].astype(str)

# Outlier removal using IQR
numeric_cols = ['Area', 'Bedrooms', 'Bathrooms', 'Floors', 'YearBuilt', 'Price']
for col in numeric_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df = df[(df[col] >= lower) & (df[col] <= upper)]
    print(f"{col}: Outliers removed. New shape: {df.shape}")

# Save cleaned data
df.to_csv("cleaned_property_data.csv", index=False)
print("Cleaned dataset saved as cleaned_property_data.csv")
