import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
import numpy as np

df = pd.read_csv("Superstore_Cleaned.csv", encoding="latin1")

print(df.head())

df["Order Date"] = pd.to_datetime(df["Order Date"])

#Create Monthly Sales
monthly_sales = (
    df
    .groupby(pd.Grouper(key="Order Date", freq="ME"))
    ["Sales"]
    .sum()
    .reset_index()
)

print(monthly_sales.head())

#Plot Monthly Sales
plt.figure(figsize=(12,6))

plt.plot(
    monthly_sales["Order Date"],
    monthly_sales["Sales"],
    marker="o"
)

plt.title("Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.grid(True)

plt.show()

#Create Time Index
monthly_sales["Month_Number"] = range(len(monthly_sales))

print(monthly_sales.head())

#Train Model
X = monthly_sales[["Month_Number"]]

y = monthly_sales["Sales"]

model = LinearRegression()

model.fit(X, y)

#Predict Existing Months
monthly_sales["Predicted_Sales"] = model.predict(X)

print(monthly_sales.head())

#Forecast Next 12 Months
future_months = pd.DataFrame({
    "Month_Number": range(
        len(monthly_sales),
        len(monthly_sales) + 12
    )
})

future_predictions = model.predict(future_months)

print(future_predictions)

#Plot Forecast
plt.figure(figsize=(12,6))

plt.plot(
    monthly_sales["Month_Number"],
    monthly_sales["Sales"],
    label="Actual Sales",
    marker="o"
)

plt.plot(
    monthly_sales["Month_Number"],
    monthly_sales["Predicted_Sales"],
    label="Trend Line",
    linewidth=3
)

plt.plot(
    future_months["Month_Number"],
    future_predictions,
    label="Forecast",
    linestyle="dashed",
    marker="o"
)

plt.title("Sales Forecast")

plt.xlabel("Month")

plt.ylabel("Sales")

plt.legend()

plt.grid(True)

plt.show()

#Save Forecast
forecast = future_months.copy()

forecast["Predicted Sales"] = future_predictions

forecast.to_csv("Sales_Forecast.csv", index=False)

print("Sales Forecast saved successfully!")
