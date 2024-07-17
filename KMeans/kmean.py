import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# 生成随机数据
X, y_true = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
# make_blobs 函数用于生成模拟的二维数据，可以指定样本数、中心数（聚类数）、每个簇的标准差等。

# 应用K均值聚类
kmeans = KMeans(n_clusters=4)
kmeans.fit(X)
y_kmeans = kmeans.predict(X)

# 可视化聚类结果
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_true, s=50, cmap='viridis')
# plt.show()
plt.savefig('kmeans_clustering.png')
plt.clf()
# 说明y_true是真实的分类标签，y_kmeans是K均值聚类算法预测的分类标签。两者可能有不同。
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, s=50, cmap='viridis')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='red', s=200, alpha=0.75, marker='X')  # 以红色十字标记聚类中心
plt.title('K-Means Clustering')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
# plt.show()
plt.savefig('kmeans_clustering_with_centers.png')