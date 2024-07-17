from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_olivetti_faces

# 加载Olivetti面部数据集
data = fetch_olivetti_faces()
images = data.images

# 选择一张图像进行去噪
image = images[10]  # 选择第十张图像

# 将图像转换为二维
original_shape = image.shape
X = image.reshape(-1)

# 应用PCA
n_components = 50  # 保留50个主成分
pca = PCA(n_components=n_components)
X_reduced = pca.fit_transform(X.reshape(1, -1))
X_recovered = pca.inverse_transform(X_reduced)

# 绘制原始和去噪后的图像
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Original Image')
ax[0].axis('off')
ax[1].imshow(X_recovered.reshape(original_shape), cmap='gray')
ax[1].set_title('Denoised Image')
ax[1].axis('off')
plt.show()
