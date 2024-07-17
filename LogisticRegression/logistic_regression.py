import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def compute_loss(y, predictions):
    return -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))

# 设置随机种子
np.random.seed(0)

# 生成二维特征的数据点
n_points = 100
x_class0 = np.random.randn(n_points, 2) + np.array([-2, -2])  # 第一类中心在(-2, -2)
x_class1 = np.random.randn(n_points, 2) + np.array([2, 2])    # 第二类中心在(2, 2)

x = np.vstack((x_class0, x_class1))
y = np.array([0]*n_points + [1]*n_points)  # 生成标签

# 将 x 数据增加一列1，用于计算截距
X_b = np.c_[np.ones((2*n_points, 1)), x]

# 参数初始化（斜率和截距）
theta = np.random.randn(3, 1)  # 包含截距的三个参数

# 学习率和迭代次数
learning_rate = 0.05
iterations = 1000

# 存储损失以便可视化
loss_history = []

# 梯度下降算法
for i in range(iterations):
    predictions = sigmoid(X_b.dot(theta))
    loss = compute_loss(y.reshape(-1, 1), predictions)
    loss_history.append(loss)
    gradients = -1 / len(X_b) * X_b.T.dot(y.reshape(-1, 1) - predictions)
    theta -= learning_rate * gradients

# 函数：根据决策边界计算x2
def plot_decision_boundary(x):
    return -(theta[1,0]*x + theta[0,0]) / theta[2,0]

# 绘制数据点和拟合的分类边界
x1_min, x1_max = x[:, 0].min() - 1, x[:, 0].max() + 1
x1_values = np.linspace(x1_min, x1_max, 400)
x2_values = plot_decision_boundary(x1_values)

plt.figure(figsize=(10, 5))
plt.scatter(x_class0[:, 0], x_class0[:, 1], color='blue', label='Class 0')
plt.scatter(x_class1[:, 0], x_class1[:, 1], color='red', label='Class 1')
plt.plot(x1_values, x2_values, color='green', label='Decision Boundary')
plt.title('Logistic Regression with 2D Features')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.show()

# 绘制损失函数随迭代次数变化的曲线
plt.figure(figsize=(10, 5))
plt.plot(loss_history, color='purple', label='Loss over iterations')
plt.title('Loss Function Over Iterations')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()
plt.show()