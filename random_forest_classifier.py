"""
文件名：random_forest_classifier.py

功能：实现随机森林分类算法，包括模型训练、参数优化、特征重要性分析和可视化

主要函数：
- load_dataset: 加载分类数据集
- train_and_evaluate_model: 训练并评估随机森林模型
- plot_confusion_matrix: 绘制混淆矩阵
- plot_feature_importance: 绘制特征重要性图
- plot_tree_structure: 绘制决策树结构图（示例树）
- plot_learning_curve: 绘制学习曲线

依赖库：
- numpy: 用于数值计算
- pandas: 用于数据处理
- sklearn: 用于机器学习模型
- matplotlib: 用于数据可视化
- os: 用于文件和目录操作
- tqdm: 用于显示进度条

其他说明：
- 支持多个数据集（breast_cancer、iris、wine）
- 包含完整的模型评估指标
- 提供特征重要性分析
- 自动保存所有可视化结果

作者：AI Assistant
创建日期：2024-03-20
修改日期：2024-03-20
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV, learning_curve
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import load_breast_cancer, load_iris, load_wine
from sklearn.tree import plot_tree
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
SAVE_PATH = 'visualizations/random_forest'
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
    训练并评估随机森林模型
    
    参数:
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标变量
    
    返回值:
        tuple: (model, X_train, X_test, y_train, y_test, y_pred)
            - model (RandomForestClassifier): 训练好的模型
            - X_train (np.ndarray): 训练集特征
            - X_test (np.ndarray): 测试集特征
            - y_train (np.ndarray): 训练集标签
            - y_test (np.ndarray): 测试集标签
            - y_pred (np.ndarray): 预测结果
    """
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 使用tqdm显示训练进度
    with tqdm(total=100, desc="Training Random Forest") as pbar:
        # 创建并训练模型
        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            random_state=42
        )
        pbar.update(20)
        
        # 模型训练
        model.fit(X_train, y_train)
        pbar.update(60)
        
        # 预测
        y_pred = model.predict(X_test)
        pbar.update(20)
    
    # 打印评估结果
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_train, X_test, y_train, y_test, y_pred

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
    
    plt.savefig(os.path.join(SAVE_PATH, 'confusion_matrix.png'))
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    绘制并保存特征重要性图
    
    参数:
        model (RandomForestClassifier): 训练好的随机森林模型
        feature_names (list): 特征名称列表
    
    返回值:
        None
    """
    importances = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importances = importances.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importances)), importances['importance'])
    plt.yticks(range(len(importances)), importances['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Random Forest Feature Importance')
    
    plt.savefig(os.path.join(SAVE_PATH, 'feature_importance.png'))
    plt.close()

def plot_tree_structure(model):
    """
    绘制并保存随机森林中的一棵决策树结构图
    
    参数:
        model (RandomForestClassifier): 训练好的随机森林模型
    
    返回值:
        None
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model.estimators_[0], filled=True, feature_names=feature_names)
    plt.title('Example Decision Tree from Random Forest')
    
    plt.savefig(os.path.join(SAVE_PATH, 'tree_structure.png'))
    plt.close()

def plot_learning_curve(model, X, y):
    """
    绘制并保存学习曲线
    
    参数:
        model (RandomForestClassifier): 训练好的随机森林模型
        X (np.ndarray): 特征矩阵
        y (np.ndarray): 目标变量
    
    返回值:
        None
    """
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=5, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training Score')
    plt.plot(train_sizes, test_mean, label='Cross-validation Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1)
    
    plt.xlabel('Training Examples')
    plt.ylabel('Score')
    plt.title('Learning Curve')
    plt.legend(loc='best')
    plt.grid(True)
    
    plt.savefig(os.path.join(SAVE_PATH, 'learning_curve.png'))
    plt.close()

if __name__ == "__main__":
    # 加载数据集
    X, y, feature_names, target_names = load_dataset('breast_cancer')
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, y_pred = train_and_evaluate_model(X, y)
    
    # 绘制并保存可视化结果
    plot_confusion_matrix(y_test, y_pred, target_names)
    plot_feature_importance(model, feature_names)
    plot_tree_structure(model)
    plot_learning_curve(model, X, y)
    
    print(f"\nAll visualizations have been saved to {SAVE_PATH} directory") 