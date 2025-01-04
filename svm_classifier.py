"""
文件名：svm_classifier.py

功能：实现支持向量机分类算法，包括模型训练、参数优化和可视化分析

主要函数：
- load_dataset: 加载分类数据集
- plot_decision_boundary_2d: 绘制二维决策边界
- plot_confusion_matrix: 绘制混淆矩阵
- plot_support_vectors: 绘制支持向量可视化

依赖库：
- numpy: 用于数值计算
- pandas: 用于数据处理
- sklearn: 用于机器学习模型
- matplotlib: 用于数据可视化
- os: 用于文件和目录操作
- tqdm: 用于显示进度条

其他说明：
- 支持多个数据集（breast_cancer、iris、wine）
- 包含完整的可视化分析功能
- 自动保存所有可视化结果

作者：AI Assistant
创建日期：2024-03-20
修改日期：2024-03-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
import os
from tqdm import tqdm

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory (str): 目录路径
    
    返回值:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# 设置可视化保存路径
SAVE_PATH = 'visualizations/svm'
ensure_dir(SAVE_PATH)

def load_dataset(dataset_name='breast_cancer'):
    """
    加载分类数据集
    
    参数:
        dataset_name (str): 数据集名称
            - 'breast_cancer': 乳腺癌数据集 (2类)
            - 'iris': 鸢尾花数据集 (3类)
            - 'wine': 红酒数据集 (3类)
    
    返回值:
        tuple: (X, y, feature_names, target_names)
            - X (np.ndarray): 特征矩阵
            - y (np.ndarray): 目标变量
            - feature_names (list): 特征名称列表
            - target_names (list): 目标类别名称列表
    """
    if dataset_name == 'breast_cancer':
        data = load_breast_cancer()
    elif dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'wine':
        data = load_wine()
    else:
        raise ValueError("Unsupported dataset name")
    
    X = data.data
    y = data.target
    feature_names = data.feature_names
    target_names = data.target_names
    
    print(f"\nLoaded {dataset_name} dataset:")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")
    print(f"Number of classes: {len(np.unique(y))}")
    
    return X, y, feature_names, target_names

def plot_decision_boundary_2d(model, X, y, feature_indices=[0, 1]):
    """
    绘制二维决策边界
    
    参数:
        model (SVC): 训练好的SVM模型
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标变量
        feature_indices (list): 要可视化的特征索引，默认[0, 1]
    
    返回值:
        None
    """
    # 选择两个特征
    X_selected = X[:, feature_indices]
    
    # 标准化
    scaler_2d = StandardScaler()
    X_scaled = scaler_2d.fit_transform(X_selected)
    
    # 创建新的SVM模型
    model_2d = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma)
    
    # 使用tqdm显示训练进度
    with tqdm(total=100, desc="Training 2D Model") as pbar:
        model_2d.fit(X_scaled, y)
        pbar.update(100)
    
    # 创建网格点
    x_min, x_max = X_scaled[:, 0].min() - 1, X_scaled[:, 0].max() + 1
    y_min, y_max = X_scaled[:, 1].min() - 1, X_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    
    # 预测网格点的类别
    Z = model_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, alpha=0.8)
    plt.xlabel(f'Feature {feature_indices[0]}')
    plt.ylabel(f'Feature {feature_indices[1]}')
    plt.title('SVM Decision Boundary (2D Projection)')
    
    plt.savefig(os.path.join(SAVE_PATH, 'svm_decision_boundary.png'))
    plt.close()

def plot_confusion_matrix(y_test, y_pred, target_names):
    """绘制混淆矩阵"""
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    tick_marks = np.arange(len(target_names))
    plt.xticks(tick_marks, target_names, rotation=45)
    plt.yticks(tick_marks, target_names)
    
    # 添加数值标签
    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, 'confusion_matrix.png'))
    plt.close()

def plot_support_vectors(model, X, y, feature_indices=[0, 1]):
    """绘制支持向量（仅使用选定的两个特征）"""
    # 选择两个特征
    X_selected = X[:, feature_indices]
    
    # 标准化
    scaler_2d = StandardScaler()
    X_scaled = scaler_2d.fit_transform(X_selected)
    
    # 创建新的SVM模型
    model_2d = SVC(kernel=model.kernel, C=model.C, gamma=model.gamma)
    model_2d.fit(X_scaled, y)
    
    # 绘制散点图
    plt.figure(figsize=(10, 8))
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, alpha=0.5)
    
    # 高亮支持向量
    sv = model_2d.support_vectors_
    plt.scatter(sv[:, 0], sv[:, 1], c='red', marker='x', s=100, 
               linewidth=2, label='Support Vectors')
    
    plt.xlabel(f'Feature {feature_indices[0]}')
    plt.ylabel(f'Feature {feature_indices[1]}')
    plt.title('Support Vectors Visualization')
    plt.legend()
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, 'support_vectors.png'))
    plt.close()

if __name__ == "__main__":
    # 加载数据集
    X, y, feature_names, target_names = load_dataset('iris')
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 创建和训练模型
    with tqdm(total=100, desc="Training SVM Model") as pbar:
        model = SVC(kernel='rbf', random_state=42)
        model.fit(X_train, y_train)
        pbar.update(100)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 模型评估
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # 可视化
    plot_decision_boundary_2d(model, X, y)
    plot_confusion_matrix(y_test, y_pred, target_names)
    plot_support_vectors(model, X, y)
    
    print(f"\nAll visualizations have been saved to {SAVE_PATH} directory") 