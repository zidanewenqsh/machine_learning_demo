"""
文件名：knn_classifier.py

功能：实现K近邻分类算法，包括模型训练、参数优化、可视化分析等功能

主要函数：
- load_dataset: 加载数据集
- train_knn_model: 训练KNN模型并进行网格搜索优化
- plot_decision_boundary_2d: 绘制二维决策边界
- plot_k_vs_accuracy: 分析K值对模型性能的影响
- plot_confusion_matrix: 绘制混淆矩阵

主要类：
- 无

依赖库：
- numpy: 用于数值计算
- pandas: 用于数据处理
- sklearn: 用于机器学习模型
- matplotlib: 用于数据可视化
- tqdm: 用于显示进度条

其他说明：
- 支持多个数据集（iris、breast_cancer、wine）
- 包含完整的可视化分析功能
- 使用GridSearchCV进行参数优化

作者：AI Assistant
创建日期：2024-03-20
修改日期：2024-03-20
"""

# K近邻分类器实现
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
import os
from tqdm import tqdm

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数：
        directory: 目录路径
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

def load_dataset(dataset_name='iris'):
    """
    加载数据集
    
    参数：
        dataset_name: str, 数据集名称
            - 'iris': 鸢尾花数据集 (3类)
            - 'breast_cancer': 乳腺癌数据集 (2类)
            - 'wine': 红酒数据集 (3类)
            
    返回：
        tuple: (特征矩阵, 目标变量, 特征名称列表, 目标类别名称列表)
    """
    if dataset_name == 'iris':
        data = load_iris()
    elif dataset_name == 'breast_cancer':
        data = load_breast_cancer()
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

def train_knn_model(X, y):
    """
    训练KNN模型并进行参数优化
    
    参数：
        X (numpy.ndarray): 特征矩阵
        y (numpy.ndarray): 目标变量
        
    返回值：
        tuple: (最佳模型, 训练集X, 测试集X, 训练集y, 测试集y, 预测结果, 标准化器)
    """
    # 使用tqdm显示训练进度
    with tqdm(total=100, desc="Training Progress") as pbar:
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pbar.update(20)
        
        # 划分训练集和测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        pbar.update(20)
        
        # 定义参数网格
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance'],
            'metric': ['euclidean', 'manhattan']
        }
        
        # 创建KNN模型并进行网格搜索
        knn = KNeighborsClassifier()
        grid_search = GridSearchCV(
            knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1
        )
        grid_search.fit(X_train, y_train)
        pbar.update(40)
        
        # 使用最佳模型进行预测
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        pbar.update(20)
    
    return best_model, X_train, X_test, y_train, y_test, y_pred, scaler

def plot_decision_boundary_2d(model, X, y, feature_indices=[0, 1], save_path='visualizations'):
    """
    绘制二维决策边界
    
    参数：
        model (KNeighborsClassifier): 训练好的KNN模型
        X (numpy.ndarray): 特征矩阵
        y (numpy.ndarray): 目标变量
        feature_indices (list): 要可视化的特征索引，默认[0, 1]
        save_path (str): 图片保存路径，默认'visualizations'
    
    返回值：
        None
    """
    ensure_dir(save_path)
    
    # 选择两个特征
    X_selected = X[:, feature_indices]
    
    # 为选定的特征创建新的标准化器
    scaler_2d = StandardScaler()
    X_scaled = scaler_2d.fit_transform(X_selected)
    
    # 创建新的KNN模型（仅使用两个特征）
    model_2d = KNeighborsClassifier(
        n_neighbors=model.n_neighbors,
        weights=model.weights,
        metric=model.metric
    )
    model_2d.fit(X_scaled, y)
    
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
    plt.title('KNN Decision Boundary (2D Projection)')
    
    # 保存图片
    plt.savefig(os.path.join(save_path, 'knn_decision_boundary.png'))
    plt.close()

def plot_k_vs_accuracy(X, y, k_range=range(1, 31), save_path='visualizations'):
    """
    绘制不同K值对应的准确率曲线并保存图片
    """
    ensure_dir(save_path)
    
    # 数据标准化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )
    
    # 计算不同k值的准确率
    train_scores = []
    test_scores = []
    
    for k in k_range:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        train_scores.append(knn.score(X_train, y_train))
        test_scores.append(knn.score(X_test, y_test))
    
    # 绘制准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, train_scores, label='Training Accuracy')
    plt.plot(k_range, test_scores, label='Testing Accuracy')
    plt.xlabel('K Value')
    plt.ylabel('Accuracy')
    plt.title('K Value vs. Model Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 保存图片
    plt.savefig(os.path.join(save_path, 'knn_k_vs_accuracy.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, target_names, save_path='visualizations'):
    """
    绘制混淆矩阵并保存图片
    """
    ensure_dir(save_path)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    
    # 添加标签
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
    plt.savefig(os.path.join(save_path, 'knn_confusion_matrix.png'))
    plt.close()

if __name__ == "__main__":
    # 设置可视化保存路径
    SAVE_PATH = 'visualizations/knn'
    ensure_dir(SAVE_PATH)
    
    print("Starting model training and evaluation...")
    
    # 加载数据集
    X, y, feature_names, target_names = load_dataset('iris')
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, y_pred, scaler = train_knn_model(X, y)
    
    # 绘制并保存决策边界
    plot_decision_boundary_2d(model, X, y, [0, 1], SAVE_PATH)
    
    # 绘制并保存K值与准确率的关系
    plot_k_vs_accuracy(X, y, save_path=SAVE_PATH)
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(y_test, y_pred, target_names, SAVE_PATH)
    
    print(f"\nAll visualization results have been saved to {SAVE_PATH} directory") 