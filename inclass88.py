import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
 
X = np.array([
    [1, 1],
    [4, 1],
    [1, 4],
    [10, 11],
    [9, 11],
    [9, 14],
    [10, 9],
    [10, 13],
    [14, 12]
])
 
plt.scatter(X[:,0], X[:,1])
plt.show()
 
# from sklearn.cluster import AgglomerativeClustering
 
# cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
# cluster.fit(X)
 
# print(cluster.labels_)
 
 
# plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow')
# plt.show()

# cluster0 = X[cluster.labels_==0]
# cluster1 = X[cluster.labels_==1]
# cluster2 = X[cluster.labels_==2]
 
# import statistics as st
 
# print("Cluster 0")
# print(st.mean(cluster0[:,0]))
# print(st.mean(cluster0[:,1]))
 
# print("Cluster 1")
# print(st.mean(cluster1[:,0]))
# print(st.mean(cluster1[:,1]))
 
# print("Cluster 2")
# print(st.mean(cluster2[:,0]))
# print(st.mean(cluster2[:,1]))

X = pd.read_csv("Downloads\SP21LWDAW7 - Sheet1.csv")
# print(data)

from sklearn.cluster import AgglomerativeClustering
 
cluster = AgglomerativeClustering(n_clusters=3, affinity='euclidean', linkage='ward')
cluster.fit(X.values)
 
print(cluster.labels_)
 
cluster0 = X[cluster.labels_==0]
cluster1 = X[cluster.labels_==1]
cluster2 = X[cluster.labels_==2]
 
import statistics as st
 
print("Cluster 0")
for i in range(6):
    print(str(round(st.mean(cluster0.iloc[:,i]), 2)) + ": " + cluster0.columns[i])
 
print("\n")
print("Cluster 1")
for i in range(6):
    print(str(round(st.mean(cluster1.iloc[:,i]), 2)) + ": " + cluster0.columns[i])
 
print("\n") 
print("Cluster 2")
for i in range(6):
    print(str(round(st.mean(cluster2.iloc[:,i]), 2)) + ": " + cluster0.columns[i])
 
# print("Cluster 1")
# print(st.mean(cluster1[:,0]))
# print(st.mean(cluster1[:,1]))
 
# print("Cluster 2")
# print(st.mean(cluster2[:,0]))
# print(st.mean(cluster2[:,1]))