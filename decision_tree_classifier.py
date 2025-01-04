"""
文件名：decision_tree_classifier.py

功能：实现决策树分类器，包括模型训练、评估和可视化。

主要函数：
- load_dataset: 加载数据集
- train_and_evaluate_model: 训练和评估模型
- visualize_tree: 可视化决策树
- plot_feature_importance: 可视化特征重要性
- plot_confusion_matrix: 可视化混淆矩阵
- perform_cross_validation: 执行交叉验证

依赖库：
- numpy
- pandas
- matplotlib
- sklearn

作者：[您的名字]
创建日期：[创建日期]
修改日期：[最后修改日期]

示例用法：
运行此文件将加载数据集，训练决策树模型，并生成相关的可视化输出。
"""

# 导入必要的库
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris, load_breast_cancer, load_wine
import os
from tqdm import tqdm
import seaborn as sns

def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory)

# 设置可视化保存路径
SAVE_PATH = 'visualizations/decision_tree'
ensure_dir(SAVE_PATH)

def load_dataset(dataset_name='iris'):
    """
    加载数据集
    可选数据集：
    - 'iris': 鸢尾花数据集 (3类)
    - 'breast_cancer': 乳腺癌数据集 (2类)
    - 'wine': 红酒数据集 (3类)
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
    print(f"Sample count: {X.shape[0]}")
    print(f"Feature count: {X.shape[1]}")
    print(f"Class count: {len(np.unique(y))}")
    
    return X, y, feature_names, target_names

def train_and_evaluate_model(X, y):
    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 创建并训练模型
    model = DecisionTreeClassifier(random_state=42)
    for i in tqdm(range(1), desc="Training Model"):
        model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 模型评估
    print("\nModel evaluation results:")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred))
    
    return model, X_train, X_test, y_train, y_test, y_pred

def visualize_tree(model, feature_names, class_names):
    """
    绘制并保存决策树结构图。
    
    参数:
    - model: 训练好的决策树模型
    - feature_names: 特征名称列表，用于标注决策树节点
    - class_names: 类别名称列表，用于标注叶子节点
    
    功能:
    - 使用sklearn的plot_tree函数可视化决策树结构
    - 将决策树结构图保存到指定的可视化目录
    """
    plt.figure(figsize=(20, 10))
    plot_tree(model, feature_names=feature_names, 
             class_names=class_names, filled=True, 
             rounded=True, fontsize=10)
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, 'decision_tree_structure.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names):
    """
    绘制并保存特征重要性图。
    
    参数:
    - model: 训练好的决策树模型，用于提取特征重要性。
    - feature_names: 特征名称列表，用于标注图中的特征。
    
    功能:
    - 此函数将模型的特征重要性绘制为水平条形图，并保存到指定路径。
    """
    importance = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    })
    importance = importance.sort_values('importance', ascending=True)
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance)), importance['importance'])
    plt.yticks(range(len(importance)), importance['feature'])
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance Ranking')
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, 'feature_importance.png'))
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    绘制并保存混淆矩阵可视化图。
    
    参数:
    - y_true: 真实标签值
    - y_pred: 模型预测的标签值
    - class_names: 类别名称列表
    
    功能:
    - 计算混淆矩阵并将其可视化
    - 将结果保存到指定的可视化目录
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, 'confusion_matrix.png'))
    plt.close()

def perform_cross_validation(model, X, y):
    """
    执行交叉验证并打印结果。
    
    参数:
    - model: 已经训练好的模型，将用于交叉验证。
    - X: 特征数据集。
    - y: 目标数据集（标签）。
    
    功能:
    - 此函数将使用提供的模型和数据执行5折交叉验证，并打印每次验证的得分、平均得分和标准差。
    """
    cv_scores = cross_val_score(model, X, y, cv=5)
    print("\n交叉验证结果:")
    print(f"CV 得分: {cv_scores}")
    print(f"平均 CV 得分: {cv_scores.mean():.4f}")
    print(f"标准差: {cv_scores.std():.4f}")

if __name__ == "__main__":
    # 加载数据集
    X, y, feature_names, target_names = load_dataset('iris')
    
    # 训练模型
    model, X_train, X_test, y_train, y_test, y_pred = train_and_evaluate_model(X, y)
    
    # 绘制并保存决策树结构
    visualize_tree(model, feature_names, target_names)
    
    # 绘制并保存特征重要性图
    plot_feature_importance(model, feature_names)
    
    # 绘制并保存混淆矩阵
    plot_confusion_matrix(y_test, y_pred, target_names)
    
    # 执行交叉验证
    perform_cross_validation(model, X, y)
    
    print(f"\nAll visualization results have been saved to {SAVE_PATH} directory") 