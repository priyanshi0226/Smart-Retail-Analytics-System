import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# ==========================
# Load RFM Table
# ==========================
rfm = pd.read_csv("RFM_Table.csv")

print("RFM Table:")
print(rfm.head())

# ==========================
# Select RFM Features
# ==========================
rfm_features = rfm[['Recency', 'Frequency', 'Monetary']]

# ==========================
# Standardize the Data
# ==========================
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm_features)

print("\nScaled Data (First 5 Rows):")
print(rfm_scaled[:5])

# ==========================
# Elbow Method
# ==========================
wcss = []

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        random_state=42,
        n_init=10
    )

    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Plot Elbow Graph
plt.figure(figsize=(8,5))

plt.plot(range(1,11), wcss, marker='o')

plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)

plt.show()

# ==========================
# K-Means Clustering
# ==========================
kmeans = KMeans(
    n_clusters=4,
    random_state=42,
    n_init=10
)

rfm["Cluster"] = kmeans.fit_predict(rfm_scaled)

# ==========================
# Cluster Centers
# ==========================
print("\nCluster Centers:")
print(kmeans.cluster_centers_)

# ==========================
# Customers in Each Cluster
# ==========================
print("\nCustomers in Each Cluster:")
print(rfm["Cluster"].value_counts().sort_index())

# ==========================
# Cluster Summary
# ==========================
cluster_summary = rfm.groupby("Cluster")[["Recency", "Frequency", "Monetary"]].mean()

print("\nCluster Summary:")
print(cluster_summary)

# ==========================
# Save Customer Segments
# ==========================
rfm.to_csv("Customer_Segments.csv", index=False)

print("\nCustomer Segmentation saved successfully!")

# ==========================
# Customer Segmentation Plot
# ==========================
plt.figure(figsize=(8,6))

scatter = plt.scatter(
    rfm["Recency"],
    rfm["Monetary"],
    c=rfm["Cluster"],
    cmap="viridis"
)

plt.xlabel("Recency")
plt.ylabel("Monetary")
plt.title("Customer Segments (K-Means Clustering)")

plt.colorbar(scatter, label="Cluster")

plt.grid(True)

plt.show()