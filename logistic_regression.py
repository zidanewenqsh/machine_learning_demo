"""
文件名：logistic_regression.py

功能：实现逻辑回归分类算法，包括模型训练、评估和可视化分析

主要函数：
- load_dataset: 加载分类数据集
- train_and_evaluate_model: 训练并评估逻辑回归模型
- plot_confusion_matrix: 绘制混淆矩阵
- plot_roc_curve: 绘制ROC曲线（二分类问题）
- plot_feature_importance: 绘制特征重要性图

依赖库：
- numpy: 用于数值计算
- pandas: 用于数据处理
- matplotlib: 用于数据可视化
- scikit-learn: 用于机器学习模型
- os: 用于文件和目录操作
- tqdm: 用于显示训练进度

其他说明：
- 支持多个数据集（breast_cancer、iris、wine）
- 包含完整的模型评估指标
- 提供多种可视化分析功能
- 自动保存所有可视化结果

作者：AI Assistant
创建日期：2024-03-20
修改日期：2024-03-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
import os
from tqdm import tqdm  # 添加tqdm库

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
SAVE_PATH = 'visualizations/logistic_regression'
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

def train_and_evaluate_model(X, y):
    """
    训练并评估逻辑回归模型
    
    参数:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标变量
    
    返回值:
        tuple: (model, X_train, X_test, y_train, y_test, y_pred, scaler)
            - model (LogisticRegression): 训练好的模型
            - X_train (np.ndarray): 训练集特征
            - X_test (np.ndarray): 测试集特征
            - y_train (np.ndarray): 训练集标签
            - y_test (np.ndarray): 测试集标签
            - y_pred (np.ndarray): 预测结果
            - scaler (StandardScaler): 标准化器
    """
    # 使用tqdm显示训练进度
    with tqdm(total=100, desc="Training Progress") as pbar:
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pbar.update(20)
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.2, random_state=42
        )
        pbar.update(20)
        
        # 创建并训练模型
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        pbar.update(40)
        
        # 预测
        y_pred = model.predict(X_test)
        pbar.update(20)
    
    # 模型评估
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_train, X_test, y_train, y_test, y_pred, scaler

def plot_confusion_matrix(y_test, y_pred, target_names):
    """
    绘制并保存混淆矩阵
    
    参数:
        y_test (np.ndarray): 真实标签
        y_pred (np.ndarray): 预测标签
        target_names (list): 目标类别名称列表
    
    返回值:
        None
    """
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

def plot_roc_curve(model, X_test, y_test):
    """
    绘制并保存ROC曲线（仅适用于二分类问题）
    
    参数:
        model (LogisticRegression): 训练好的模型
        X_test (np.ndarray): 测试集特征
        y_test (np.ndarray): 测试集标签
    
    返回值:
        None
    """
    if len(np.unique(y_test)) != 2:
        print("ROC curve is only applicable for binary classification")
        return
    
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, 'roc_curve.png'))
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    绘制并保存特征重要性图
    
    参数:
        model (LogisticRegression): 训练好的模型
        feature_names (list): 特征名称列表
    
    返回值:
        None
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': abs(model.coef_[0]) if len(model.classes_) == 2 
                     else np.mean(abs(model.coef_), axis=0)
    })
    importance = importance.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance['importance'])
    plt.yticks(range(len(importance)), importance['feature'])
    plt.xlabel('Feature Importance (Absolute Coefficients)')
    plt.title('Feature Importance Ranking')
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, 'feature_importance.png'))
    plt.close()

if __name__ == "__main__":
    # 加载数据集
    X, y, feature_names, target_names = load_dataset('breast_cancer')
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, y_pred, scaler = train_and_evaluate_model(X, y)
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(y_test, y_pred, target_names)
    
    # 绘制并保存ROC曲线（仅适用于二分类问题）
    plot_roc_curve(model, X_test, y_test)
    
    # 绘制并保存特征重要性图
    plot_feature_importance(model, feature_names)
    
    print(f"\nAll visualizations have been saved to {SAVE_PATH} directory") 