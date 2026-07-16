import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# ============================================
# Load Dataset
# ============================================

df = pd.read_csv("Superstore_Cleaned.csv", encoding="latin1")

print("Dataset Preview:")
print(df.head())

# ============================================
# Keep Required Columns
# ============================================

basket = df[['Order ID', 'Sub-Category']]

print("\nBasket Preview:")
print(basket.head())

# ============================================
# Check Products per Order
# ============================================

basket_size = df.groupby("Order ID")["Sub-Category"].count()

print("\nProducts Per Order Statistics:")
print(basket_size.describe())

print("\nOrders with More Than One Product:")
print((basket_size > 1).sum())

# ============================================
# Create Basket Matrix
# ============================================

basket = (
    basket
    .groupby(["Order ID", "Sub-Category"])["Sub-Category"]
    .count()
    .unstack()
    .fillna(0)
)

# Convert Counts to Boolean
basket = basket.astype(bool)

print("\nBasket Shape:")
print(basket.shape)

print("\nBasket Preview:")
print(basket.head())

# ============================================
# Generate Frequent Itemsets
# ============================================

frequent_items = apriori(
    basket,
    min_support=0.01,
    use_colnames=True
)

print("\nFrequent Itemsets:")
print(frequent_items.head())

print("\nNumber of Frequent Itemsets:")
print(len(frequent_items))

# ============================================
# Itemset Length
# ============================================

frequent_items["Length"] = frequent_items["itemsets"].apply(len)

print("\nItemset Length Distribution:")
print(frequent_items["Length"].value_counts())

# ============================================
# Generate Association Rules
# ============================================

rules = association_rules(
    frequent_items,
    metric="confidence",
    min_threshold=0.20
)

print("\nAssociation Rules:")

if len(rules) > 0:

    rules = rules.sort_values(by="lift", ascending=False)

    print(
        rules[
            [
                "antecedents",
                "consequents",
                "support",
                "confidence",
                "lift"
            ]
        ].head(20)
    )

else:
    print("No association rules found.")

print("\nNumber of Association Rules:")
print(len(rules))

# ============================================
# Save Results
# ============================================

frequent_items.to_csv("Frequent_Itemsets.csv", index=False)
rules.to_csv("Association_Rules.csv", index=False)

print("\nFrequent Itemsets saved successfully!")
print("Association Rules saved successfully!")