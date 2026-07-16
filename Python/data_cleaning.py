import pandas as pd

df = pd.read_csv("SampleSuperstore.csv", encoding="cp1252")


# Display first 5 rows
print(df.head())

# Dataset information
print("\nDataset Information:")
print(df.info())

# Shape of dataset
print("\nShape of Dataset:")
print(df.shape)

# Missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate records
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Data types
print("\nData Types:")
print(df.dtypes)

# Statistical Summary
print("\nStatistical Summary:")
print(df.describe())



# Check Missing Values
print("Missing Values:")
print(df.isnull().sum())

# Check Duplicate Rows
print("\nDuplicate Records:")
print(df.duplicated().sum())

# Remove Duplicate Rows (if any)
df = df.drop_duplicates()

# Convert Date Columns
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])

# Check Data Types
print("\nUpdated Data Types:")
print(df.dtypes)

# Display Dataset Information
print("\nDataset Information:")
print(df.info())


# Shipping Duration
df["Shipping Duration"] = (df["Ship Date"] - df["Order Date"]).dt.days
print(df[["Order Date", "Ship Date", "Shipping Duration"]].head())

# Profit Margin (%)
df["Profit Margin (%)"] = (df["Profit"] / df["Sales"]) * 100

# Order Year
df["Order Year"] = df["Order Date"].dt.year

# Order Month
df["Order Month"] = df["Order Date"].dt.month_name()

# Order Quarter
df["Order Quarter"] = df["Order Date"].dt.quarter

print(df[[
    "Shipping Duration",
    "Profit Margin (%)",
    "Order Year",
    "Order Month",
    "Order Quarter"
]].head())

# Sales Category
df["Sales Category"] = pd.cut(
    df["Sales"],
    bins=[0, 100, 500, float("inf")],
    labels=["Low", "Medium", "High"]
)

print("Number of rows:", len(df))
print("Shape:", df.shape)

# Save Cleaned Dataset
df.to_csv("Superstore_Cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")
