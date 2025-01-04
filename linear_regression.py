"""
文件名：linear_regression.py

功能：实现线性回归算法，包括模型训练、预测和评估

主要函数：
- load_dataset: 加载回归数据集（加利福尼亚房价或糖尿病数据集）
- train_and_evaluate_model: 训练线性回归模型并评估性能
- plot_prediction_vs_actual: 可视化预测值与实际值的对比
- plot_residuals: 可视化残差分布
- plot_feature_importance: 可视化特征重要性

依赖库：
- numpy: 用于数值计算
- pandas: 用于数据处理
- matplotlib: 用于数据可视化
- scikit-learn: 用于机器学习模型
- os: 用于文件和目录操作
- tqdm: 用于显示进度条

其他说明：
- 支持多个sklearn内置数据集
- 提供完整的模型评估指标
- 包含多种可视化方法
- 自动保存所有可视化结果

作者：AI Assistant
创建日期：2024-03-20
修改日期：2024-03-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.datasets import fetch_california_housing, load_diabetes
import os
from tqdm import tqdm

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory (str): 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# 设置可视化保存路径
SAVE_PATH = 'visualizations/linear_regression'
ensure_dir(SAVE_PATH)

def load_dataset(dataset_name='california'):
    """
    加载回归数据集
    
    参数:
        dataset_name (str): 数据集名称，可选 'california' 或 'diabetes'
    
    返回值:
        tuple: (X, y, feature_names)
            - X: 特征矩阵
            - y: 目标变量
            - feature_names: 特征名称列表
    """
    if dataset_name == 'california':
        data = fetch_california_housing()
    elif dataset_name == 'diabetes':
        data = load_diabetes()
    else:
        raise ValueError("Unsupported dataset name")
    
    X = data.data
    y = data.target
    feature_names = data.feature_names
    
    print(f"Dataset loaded: {dataset_name}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    
    return X, y, feature_names

def train_and_evaluate_model(X, y):
    """
    训练并评估线性回归模型
    
    参数:
        X (np.ndarray): 特征数据
        y (np.ndarray): 目标数据
    
    返回值:
        tuple: (model, X_train, X_test, y_train, y_test, y_pred)
            - model: 训练好的模型
            - X_train: 训练数据特征
            - X_test: 测试数据特征
            - y_train: 训练数据目标
            - y_test: 测试数据目标
            - y_pred: 模型预测结果
    """
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 初始化模型
    model = LinearRegression()
    
    # 使用tqdm显示训练进度
    epochs = 10  # 假设我们模拟10次迭代训练过程
    for i in tqdm(range(epochs), desc="Training Progress"):
        model.fit(X_train, y_train)  # 训练模型
    
    # 预测测试集结果
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    r2 = r2_score(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    
    print(f"R^2 Score: {r2}")
    print(f"Mean Squared Error: {mse}")
    
    return model, X_train, X_test, y_train, y_test, y_pred

def plot_prediction_vs_actual(y_test, y_pred):
    """
    绘制预测值与实际值的对比图
    
    参数:
        y_test (np.ndarray): 测试集的实际值
        y_pred (np.ndarray): 模型预测的值
    
    返回值:
        None
    """
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Predicted vs Actual Values')
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, 'prediction_vs_actual.png'))
    plt.close()

def plot_residuals(y_test, y_pred):
    """
    绘制残差分布图
    
    参数:
        y_test (np.ndarray): 测试集的实际值
        y_pred (np.ndarray): 模型预测的值
    
    返回值:
        None
    """
    residuals = y_test - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title('Residuals Distribution')
    plt.axhline(y=0, color='r', linestyle='--')
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, 'residuals.png'))
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    可视化并保存模型的特征重要性
    
    参数:
        model (LinearRegression): 训练好的模型
        feature_names (list): 特征名称列表
    
    返回值:
        None
    """
    plt.figure(figsize=(10, 8))
    importance = np.abs(model.coef_)
    # 按重要性降序排序
    indices = np.argsort(importance)[::-1]
    plt.bar(range(len(indices)), importance[indices], align='center')
    plt.xticks(range(len(indices)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Feature Importance (Absolute Coefficients)')  # 英文
    plt.title('Feature Importance Ranking')  # 英文
    plt.savefig(os.path.join(SAVE_PATH, 'feature_importance.png'))
    plt.close()

if __name__ == "__main__":
    # 加载数据集
    X, y, feature_names = load_dataset('california')
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, y_pred = train_and_evaluate_model(X, y)
    
    # 绘制并保存预测值与实际值的对比图
    plot_prediction_vs_actual(y_test, y_pred)
    
    # 绘制并保存残差图
    plot_residuals(y_test, y_pred)
    
    # 绘制并保存特征重要性图
    plot_feature_importance(model, feature_names)
    
    print("All visualizations have been saved to", SAVE_PATH) 