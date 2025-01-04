# 机器学习基础教程

## 数学基础
- 矩阵乘法
- 计算特征值
- 矩阵分解
- 概率分布
- 假设检验
- 计算导数
- 梯度计算
- 积分计算

## 编程基础
- 创建NumPy数组
- 操作Pandas数据框
- 定义Python函数
- 导入Python模块
- Python列表推导
- Python字典操作
- Python集合操作
- Python文件读取

## 机器学习基础
- 线性回归模型
- 逻辑回归模型
- 决策树分类器
- SVM分类器
- K近邻算法
- 随机森林分类器
- 梯度提升分类器
- 贝叶斯分类器
- 一个隐藏层的神经网络
- 互信息特征选择
- 标准化特征缩放
- 归一化特征缩放
- 网格搜索超参数调优
- 随机搜索超参数调优
- 贝叶斯优化超参数调优
- 准确率计算
- 精确率计算
- 召回率计算
- F1分数计算
- 混淆矩阵可视化
- ROC曲线可视化
- AUC分数计算

## 无监督学习
- K均值聚类
- 层次聚类
- DBSCAN聚类
- 高斯混合模型
- PCA降维
- LDA降维
- t-SNE降维
- 轮廓系数聚类评估
- 戴维森-鲍德温指数聚类评估
- 卡林斯基-哈拉巴斯指数聚类评估

## 强化学习基础
- Q学习算法
- 蒙特卡洛方法
- 深度Q网络(DQN)
- 策略梯度方法
- 演员-评论家方法
- 优势演员-评论家(A2C)方法
- 近端策略优化(PPO)方法
- 深度确定性策略梯度(DDPG)方法
- 双延迟深度确定性策略梯度(TD3)方法
- 软演员-评论家(SAC)方法

## 实战应用
- 数据预处理流程
- 模型训练评估流程
- 使用Flask部署模型
- 使用Django部署模型
- 使用FastAPI部署模型
- 使用Streamlit部署模型
- 使用Docker部署模型
- 使用Kubernetes部署模型
- 使用MLflow监控模型
- 使用TensorBoard监控模型
- 使用Wandb监控模型
- 使用DVC进行模型版本控制
- 使用Git-LFS进行模型版本控制
- 使用MLflow进行模型版本控制
- 使用TensorBoard进行模型版本控制
- 使用Wandb进行模型版本控制
- 使用LIME解释模型
- 使用SHAP解释模型
- 使用ELI5解释模型
- 使用Interpret解释模型
- 使用MLxtend解释模型
- 使用H2O解释模型
- 使用Alibi解释模型
- 使用Captum解释模型
- 使用AIX360解释模型
- 使用What-If-Tool解释模型
- 使用ModelDB解释模型
- 使用ModelX解释模型
- 使用ModelHub解释模型
- 使用ModelScope解释模型
- 使用ModelStore解释模型
- 使用ModelTracker解释模型
- 使用ModelZoo解释模型
- 使用ModelGarden解释模型
- 使用ModelHubAI解释模型
- 使用ModelScopeAI解释模型
- 使用ModelStoreAI解释模型
- 使用ModelTrackerAI解释模型
- 使用ModelZooAI解释模型
- 使用ModelGardenAI解释模型
- 使用ModelHubAI解释模型
- 使用ModelScopeAI解释模型

## 算法Python文件总结

### matrix_operations.py
- 矩阵运算实现
  - 矩阵乘法（NumPy dot运算和@运算符）
  - 特征值和特征向量计算
  - 矩阵分解（LU分解）
  - 矩阵可视化（使用Matplotlib）
- 数据集：自定义矩阵示例

### linear_regression.py
- 线性回归实现
  - 使用sklearn内置数据集
  - 数据预处理和划分
  - 模型训练与预测
  - 模型评估（R²分数、MSE）
  - 可视化：预测结果对比图和残差分布图
- 数据集：可选sklearn内置数据集

### logistic_regression.py
- 逻辑回归实现
  - 使用sklearn内置数据集
  - 数据标准化和划分
  - 模型训练与预测
  - 模型评估（准确率、分类报告）
  - 可视化：决策边界、混淆矩阵、ROC曲线
- 数据集：breast_cancer, iris, wine

### decision_tree_classifier.py
- 决策树分类器实现
  - 使用sklearn内置数据集
  - 数据预处理和划分
  - 模型训练（含参数设置）
  - 模型评估（准确率、分类报告）
  - 可视化：决策树结构、决策边界、特征重要性
  - 交叉验证评估
- 数据集：iris, breast_cancer, wine

### svm_classifier.py
- 支持向量机分类器实现
  - 数据集选择和预处理
  - 网格搜索参数优化
  - 多种核函数支持
  - 模型评估（准确率、分类报告）
  - 可视化：二维决策边界、混淆矩阵
  - 支持向量统计
- 数据集：breast_cancer, iris, wine
