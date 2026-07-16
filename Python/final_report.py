import pandas as pd
sales = pd.read_csv("Superstore_Cleaned.csv", encoding="latin1")

customers = pd.read_csv("Customer_Segments.csv")

forecast = pd.read_csv("Sales_Forecast.csv")

rules = pd.read_csv("Association_Rules.csv")

#Overall KPIs
total_sales = sales["Sales"].sum()

total_profit = sales["Profit"].sum()

total_orders = sales["Order ID"].nunique()

total_customers = sales["Customer ID"].nunique()

average_order = sales["Sales"].mean()

#Best Category
best_category = (
    sales.groupby("Category")["Sales"]
    .sum()
    .idxmax()
)

best_category_sales = (
    sales.groupby("Category")["Sales"]
    .sum()
    .max()
)

#Best Region
best_region = (
    sales.groupby("Region")["Sales"]
    .sum()
    .idxmax()
)

best_region_sales = (
    sales.groupby("Region")["Sales"]
    .sum()
    .max()
)

#Customer Segmentation Summary
cluster_summary = (
    customers
    .groupby("Cluster")[["Recency","Frequency","Monetary"]]
    .mean()
)

print(cluster_summary)

#Best Association Rule
best_rule = rules.sort_values(
    by="lift",
    ascending=False
).iloc[0]

print(best_rule)

#Forecast Summary
average_future_sales = forecast["Predicted Sales"].mean()

highest_future_sales = forecast["Predicted Sales"].max()

#Create Final Report
report = pd.DataFrame({

    "Metric":[
        "Total Sales",
        "Total Profit",
        "Total Orders",
        "Total Customers",
        "Average Order Value",
        "Best Category",
        "Category Sales",
        "Best Region",
        "Region Sales",
        "Average Forecast Sales",
        "Highest Forecast Sales"
    ],

    "Value":[
        total_sales,
        total_profit,
        total_orders,
        total_customers,
        average_order,
        best_category,
        best_category_sales,
        best_region,
        best_region_sales,
        average_future_sales,
        highest_future_sales
    ]

})

print(report)

#Save Report
report.to_csv("Final_Project_Report.csv", index=False)

print("\nFinal Project Report saved successfully!")

#Print Business Insights
print("\n========== BUSINESS INSIGHTS ==========")

print(f"Highest Revenue Category : {best_category}")

print(f"Highest Revenue Region : {best_region}")

print(f"Total Customers : {total_customers}")

print(f"Total Orders : {total_orders}")

print(f"Average Order Value : {average_order:.2f}")

print(f"Average Forecast Sales : {average_future_sales:.2f}")

print("\nBest Cross-Selling Rule:")

print(
    f"{best_rule['antecedents']} --> {best_rule['consequents']}"
)

print(f"Confidence : {best_rule['confidence']:.2f}")

print(f"Lift : {best_rule['lift']:.2f}")

print("\n======================================")