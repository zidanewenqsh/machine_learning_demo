import numpy as np
import matplotlib.pyplot as plt

# 生成一些模拟数据
np.random.seed(0)
x = 2 * np.random.rand(100, 1)
y = 4 + 3 * x + np.random.randn(100, 1)

# 将 x 数据增加一列1，用于计算截距
X_b = np.c_[np.ones((100, 1)), x]  # 添加 x0 = 1 到每个实例

# 参数初始化（斜率和截距）
theta = np.random.randn(2, 1)  # 随机初始化

# 学习率和迭代次数
learning_rate = 0.01
iterations = 1000

# 存储损失以便可视化
loss_history = []

# 梯度下降算法
for i in range(iterations):
    gradients = -2 / len(X_b) * X_b.T.dot(y - X_b.dot(theta))
    theta -= learning_rate * gradients
    loss = np.sum((y - X_b.dot(theta))**2) / len(y)
    loss_history.append(loss)

# 计算最终的预测值
y_pred = X_b.dot(theta)

# 绘制数据点和拟合直线
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='blue', label='Data points')
plt.plot(x, y_pred, color='red', label='Fitted line - Gradient Descent with Matrix')
plt.title('Linear Fit using Matrix Representation')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

# 绘制损失函数随迭代次数变化的曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_history, color='green', label='Loss over iterations')
plt.title('Loss Function Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()
