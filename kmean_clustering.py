import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
df = pd.read_csv(r"C:\Users\VIAJAYLAXMI\OneDrive\Desktop\python\Mall_Customers.csv")
print("First 5 Rows of Dataset:")
print(df.head())
print("\nDataset Information:")
print(df.info())
print("\nMissing Values:")
print(df.isnull().sum())
X = df.iloc[:, [3, 4]].values
wcss = []
for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init='k-means++',
        random_state=42,
        n_init=10
    )
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.grid(True)
plt.show()
kmeans = KMeans(
    n_clusters=5,
    init='k-means++',
    random_state=42,
    n_init=10
)
y_kmeans = kmeans.fit_predict(X)
df["Cluster"] = y_kmeans
print("\nClustered Dataset:")
print(df.head())
plt.figure(figsize=(8,6))
colors = ['red', 'blue', 'green', 'cyan', 'magenta']
for i in range(5):
    plt.scatter(
        X[y_kmeans == i, 0],
        X[y_kmeans == i, 1],
        s=70,
        c=colors[i],
        label=f'Cluster {i+1}'
    )
plt.scatter(
    kmeans.cluster_centers_[:,0],
    kmeans.cluster_centers_[:,1],
    s=300,
    c='yellow',
    marker='*',
    edgecolors='black',
    label='Centroids'
)
plt.title("Customer Segmentation using K-Means")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.legend()
plt.grid(True)
plt.show()
df.to_csv("Mall_Customers_Clustered.csv", index=False)
print("\nClustered dataset saved as 'Mall_Customers_Clustered.csv'")
print("\nCluster Centers:")
print(kmeans.cluster_centers_)
print("\nCustomers in Each Cluster:")
print(df["Cluster"].value_counts().sort_index())