import pandas as pd
import numpy as np

df = pd.read_csv("Superstore_Cleaned.csv", encoding="latin1")

#Choose a Reference Date
df["Order Date"] = pd.to_datetime(df["Order Date"])
reference_date = df["Order Date"].max() + pd.Timedelta(days=1)
print(reference_date)

#Create the RFM Table
rfm = df.groupby("Customer ID").agg({
    "Order Date": lambda x: (reference_date - x.max()).days,
    "Order ID": "nunique",
    "Sales": "sum"
})

#Rename Columns
rfm.columns = ["Recency", "Frequency", "Monetary"]

#Display Results
print(rfm.head())

#Check Dataset Information
print(rfm.info())
print(rfm.describe())

#Save the RFM Table
rfm.to_csv("RFM_Table.csv")
print("RFM Table saved successfully!")