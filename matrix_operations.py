"""
文件名：matrix_operations.py

功能：实现矩阵运算相关功能，包括矩阵乘法、特征值计算和矩阵可视化

主要函数：
- matrix_multiplication: 实现矩阵乘法示例
- eigenvalue_calculation: 实现特征值和特征向量计算
- visualize_matrix: 实现矩阵可视化并保存结果

依赖库：
- numpy: 用于矩阵运算
- matplotlib: 用于数据可视化
- os: 用于文件和目录操作

其他说明：
- 支持任意大小的矩阵运算
- 提供矩阵可视化功能
- 自动保存可视化结果

作者：AI Assistant
创建日期：2024-03-20
修改日期：2024-03-20
"""

import numpy as np
import matplotlib.pyplot as plt
import os

def ensure_dir(directory):
    """
    确保目录存在，如果不存在则创建
    
    参数:
        directory (str): 目录路径，用于检查和创建
    
    返回值:
        None
    """
    if not os.path.exists(directory):
        os.makedirs(directory)

# 设置可视化保存路径
SAVE_PATH = 'visualizations/matrix_operations'
ensure_dir(SAVE_PATH)

def matrix_multiplication():
    """
    矩阵乘法示例函数
    
    功能:
        1. 创建两个示例矩阵
        2. 使用两种不同的方法进行矩阵乘法运算
        3. 打印计算结果进行对比
    
    参数:
        无
    
    返回值:
        None
    
    示例:
        >>> matrix_multiplication()
        Matrix A:
        [[1 2]
         [3 4]]
        Matrix B:
        [[5 6]
         [7 8]]
        Matrix Multiplication Result (A × B):
        [[19 22]
         [43 50]]
    """
    # 创建示例矩阵
    A = np.array([[1, 2], [3, 4]])
    B = np.array([[5, 6], [7, 8]])
    
    # 使用 NumPy 矩阵乘法
    C1 = np.dot(A, B)
    # 使用 @ 运算符
    C2 = A @ B
    
    print("Matrix A:")
    print(A)
    print("\nMatrix B:")
    print(B)
    print("\nMatrix Multiplication Result (A × B):")
    print(C1)

def eigenvalue_calculation():
    """
    特征值和特征向量计算示例函数
    
    功能:
        1. 创建一个2x2示例矩阵
        2. 计算该矩阵的特征值和特征向量
        3. 打印计算结果
    
    参数:
        无
    
    返回值:
        None
    
    说明:
        使用 numpy.linalg.eig 函数进行计算
    """
    # 创建示例矩阵
    A = np.array([[4, -2], [1, 1]])
    
    # 计算特征值和特征向量
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    print("Matrix for eigenvalue calculation:")
    print(A)
    print("\nEigenvalues:")
    print(eigenvalues)
    print("\nEigenvectors:")
    print(eigenvectors)

def visualize_matrix(matrix, title, save_name):
    """
    矩阵可视化并保存结果
    
    功能:
        1. 将输入矩阵可视化为热力图
        2. 在每个单元格中显示具体数值
        3. 添加颜色条显示数值范围
        4. 保存可视化结果到指定路径
    
    参数:
        matrix (np.ndarray): 要可视化的矩阵，支持任意大小的二维数组
        title (str): 图表标题，将显示在图表顶部
        save_name (str): 保存的文件名，不需要包含文件扩展名
    
    返回值:
        None
    
    注意:
        - 保存的文件格式为PNG
        - 保存路径由全局变量 SAVE_PATH 指定
        - 矩阵中的数值将保留两位小数显示
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(matrix, cmap='viridis')
    plt.colorbar()
    plt.title(title)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            plt.text(j, i, f'{matrix[i, j]:.2f}',
                    ha='center', va='center')
    
    # 保存图片
    plt.savefig(os.path.join(SAVE_PATH, f'{save_name}.png'))
    plt.close()

if __name__ == "__main__":
    # 执行矩阵乘法
    print("=== Matrix Multiplication ===")
    matrix_multiplication()
    
    # 计算特征值
    print("\n=== Eigenvalue Calculation ===")
    eigenvalue_calculation()
    
    # 创建并可视化随机矩阵
    random_matrix = np.random.rand(5, 5)
    visualize_matrix(random_matrix, "Random Matrix Visualization", "random_matrix")
    
    # 创建并可视化相关性矩阵
    corr_matrix = np.corrcoef(np.random.rand(5, 100))
    visualize_matrix(corr_matrix, "Correlation Matrix", "correlation_matrix")
    
    print(f"\nAll visualizations have been saved to {SAVE_PATH} directory") 