import pandas as pd

sales = pd.read_csv("Superstore_Cleaned.csv", encoding="latin1")
customers = pd.read_csv("Customer_Segments.csv")
forecast = pd.read_csv("Sales_Forecast.csv")
rules = pd.read_csv("Association_Rules.csv")

print("Sales Dataset:", sales.shape)
print("Customer Segments:", customers.shape)
print("Sales Forecast:", forecast.shape)
print("Association Rules:", rules.shape)

#Sales KPIs
kpi = pd.DataFrame({

    "Total Sales":[sales["Sales"].sum()],

    "Total Profit":[sales["Profit"].sum()],

    "Total Orders":[sales["Order ID"].nunique()],

    "Total Customers":[sales["Customer ID"].nunique()],

    "Average Order Value":[sales["Sales"].mean()]

})

print(kpi)

#Category Sales
category_sales = (

    sales
    .groupby("Category")[["Sales","Profit"]]
    .sum()
    .reset_index()

)

print(category_sales)

#Region Sales
region_sales = (

    sales
    .groupby("Region")[["Sales","Profit"]]
    .sum()
    .reset_index()

)

print(region_sales)

#Customer Cluster Summary
cluster_summary = (

    customers
    .groupby("Cluster")
    [["Recency","Frequency","Monetary"]]
    .mean()
    .reset_index()

)

print(cluster_summary)

#Top 10 Association Rules
top_rules = (

    rules
    .sort_values("lift", ascending=False)
    .head(10)

)

print(top_rules[
    [
        "antecedents",
        "consequents",
        "confidence",
        "lift"
    ]
])

#Forecast Preview
print(forecast.head())

#Save Dashboard Files
kpi.to_csv("Dashboard_KPIs.csv", index=False)

category_sales.to_csv("Dashboard_Category.csv", index=False)

region_sales.to_csv("Dashboard_Region.csv", index=False)

cluster_summary.to_csv("Dashboard_Clusters.csv", index=False)

top_rules.to_csv("Dashboard_Rules.csv", index=False)

print("\nDashboard datasets created successfully!")