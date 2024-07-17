import numpy as np
from sklearn.decomposition import PCA

# 生成一些高维数据
np.random.seed(42)
X = np.random.rand(100, 1000)  # 100个样本，每个样本1000个特征

# 应用PCA进行数据压缩
pca = PCA(n_components=100)  # 压缩到100个主成分
X_compressed = pca.fit_transform(X)
print("Original shape:", X.shape)
print("Compressed shape:", X_compressed.shape)
